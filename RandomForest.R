library(randomForestSRC)
library(survival)
library(survcomp)


train_data = read.csv("/home/UT_shared/data/train.csv", header = TRUE)
test_data = read.csv("/home/UT_shared/data/test.csv", header = TRUE)

model <- rfsrc(Surv(year, status) ~ ., data = train_data,
               ntree = 10,
               splitrule = "logrankCR",  # for competing risks
               cause = 1)

time_grid <- model$time.interest
B <- 20
bootstrap_cindex <- numeric(B)

for (b in 1:B) {
  cat(sprintf("Processing bootstrap %d...\n", b))
  
  test_path <- sprintf("/home/UT_shared/data/bootstrap500/bootstrap_%d.csv", b)
  test_data <- read.csv(test_path, header = TRUE)
  
  # Quick sanity check (optional)
  # cat("  n =", nrow(test_data), ", events cause1 =", sum(test_data$status == 1), "\n")
  
  pred <- predict(model, newdata = test_data)
  cif_matrix <- pred$cif[, , "CIF.1"]
  
  cindex_list <- numeric(length(time_grid))
  
  for (i in seq_along(time_grid)) {
    cif_at_t <- cif_matrix[, i]
    
    concord <- try(
      concordance.index(
        x          = -cif_at_t,
        surv.time  = test_data$year,         # use 'year'
        surv.event = test_data$status == 1,  # cause 1
        method     = "noether"
      ),
      silent = TRUE
    )
    
    cindex_list[i] <- if (!inherits(concord, "try-error")) concord$c.index else NA
  }
  
  cindex_valid <- cindex_list[!is.na(cindex_list)]
  bootstrap_cindex[b] <- if (length(cindex_valid) > 0) mean(cindex_valid) else NA
}

bootstrap_cindex_clean <- bootstrap_cindex[!is.na(bootstrap_cindex)]
if (length(bootstrap_cindex_clean) == 0) {
  stop("All bootstrap C-indices are NA; check concordance.index inputs.")
}

mean_cindex <- mean(bootstrap_cindex_clean)
se <- sd(bootstrap_cindex_clean) / sqrt(length(bootstrap_cindex_clean))
ci <- mean_cindex + c(-1, 1) * qt(0.975, df = length(bootstrap_cindex_clean) - 1) * se

cat(sprintf("\nC-index summary over %d bootstraps:\n", length(bootstrap_cindex_clean)))
cat(sprintf("Mean time-dependent C-index: %.4f\n", mean_cindex))
cat(sprintf("95%% CI: %.4f – %.4f\n", ci[1], ci[2]))


## 3. Single test set: time-dependent C-index and at specific time
# Re-read original test set
test_data <- read.csv("/home/UT_shared/data/test.csv", header = TRUE)
pred <- predict(model, newdata = test_data)

if (!is.null(dimnames(pred$cif)[[3]]) && "CIF.1" %in% dimnames(pred$cif)[[3]]) {
  cif_matrix <- pred$cif[, , "CIF.1"]
} else {
  cif_matrix <- pred$cif[, , 1]
}

time_grid <- model$time.interest
cindex_list <- numeric(length(time_grid))

for (i in seq_along(time_grid)) {
  cif_at_t <- cif_matrix[, i]
  
  concord <- try(
    concordance.index(
      x          = -cif_at_t,
      surv.time  = test_data$year,
      surv.event = test_data$status == 1,
      method     = "noether"
    ),
    silent = TRUE
  )
  
  cindex_list[i] <- if (!inherits(concord, "try-error")) concord$c.index else NA
}

cindex_valid <- cindex_list[!is.na(cindex_list)]
mean_cindex_test <- mean(cindex_valid)

cat(sprintf("Mean time-dependent C-index over %d time points (original test): %.4f\n",
            length(cindex_valid), mean_cindex_test))

# C-index at a specific time point (e.g., t = 24)
target_time <- 24
closest_time_idx <- which.min(abs(time_grid - target_time))
cif_cause1_at_t <- cif_matrix[, closest_time_idx]

cindex_result <- concordance.index(
  x          = -cif_cause1_at_t,
  surv.time  = test_data$year,
  surv.event = test_data$status == 1,
  method     = "noether"
)

cat(sprintf("C-index for cause 1 at time %d: %.4f\n",
            target_time, cindex_result$c.index))


calc_cr_brier <- function(times, status, pred_cif, time_grid, cause_of_interest = 1) {
  # Estimate Censoring Distribution G(t)
  cens_fit <- survfit(Surv(times, status == 0) ~ 1)
  
  get_G <- function(t_query) {
    s <- summary(cens_fit, times = t_query, extend = TRUE)
    s$surv
  }
  
  brier_scores <- numeric(length(time_grid))
  
  for (j in seq_along(time_grid)) {
    t_eval <- time_grid[j]
    pred <- pred_cif[, j]
    
    # 1. Did the event happen by time t? (Truth)
    outcome <- as.numeric((times <= t_eval) & (status == cause_of_interest))
    
    # 2. Squared Error
    diff_sq <- (outcome - pred)^2
    
    # 3. IPCW Weights
    weights <- numeric(length(times))
    
    # A: Event happened (observed) -> Weight = 1/G(T_i)
    mask_event <- (times <= t_eval) & (status != 0)
    if (any(mask_event)) weights[mask_event] <- 1 / get_G(times[mask_event])
    
    # B: Censored AFTER t_eval (still in study) -> Weight = 1/G(t_eval)
    mask_surv <- (times > t_eval)
    if (any(mask_surv)) weights[mask_surv] <- 1 / get_G(rep(t_eval, sum(mask_surv)))
    
    # C: Censored BEFORE t_eval -> Weight = 0 (default)
    
    # Avoid division by zero issues
    weights[is.infinite(weights)] <- 0 
    
    brier_scores[j] <- mean(weights * diff_sq)
  }
  return(brier_scores)
}

calc_ibs <- function(brier_scores, time_grid) {
  dt <- diff(time_grid)
  avg_h <- (brier_scores[-1] + brier_scores[-length(brier_scores)]) / 2
  integral <- sum(dt * avg_h)
  return(integral / (max(time_grid) - min(time_grid)))
}

bootstrap_ibs <- numeric(B)

cat(sprintf("Starting IBS evaluation on %d bootstraps...\n", B))

for (b in 1:B) {
  # 1. Load Data
  test_path <- sprintf("/home/UT_shared/data/bootstrap500/bootstrap_%d.csv", b)
  if(!file.exists(test_path)) {
    cat(sprintf("Warning: File not found %s, skipping...\n", test_path))
    bootstrap_ibs[b] <- NA
    next
  }
  
  test_data <- read.csv(test_path, header = TRUE)
  
  # 2. Predict
  pred <- predict(model, newdata = test_data)
  
  # Extract CIF for Cause 1
  if (!is.null(dimnames(pred$cif)[[3]]) && "CIF.1" %in% dimnames(pred$cif)[[3]]) {
    cif_matrix <- pred$cif[, , "CIF.1"]
  } else {
    cif_matrix <- pred$cif[, , 1]
  }
  
  # 3. Calculate Brier Score Vector (Value at every time point)
  bs_vector <- try(
    calc_cr_brier(
      times = test_data$year,
      status = test_data$status,
      pred_cif = cif_matrix,
      time_grid = time_grid,
      cause_of_interest = 1
    ), 
    silent = TRUE
  )
  
  # 4. Integrate to get IBS
  if (!inherits(bs_vector, "try-error") && !any(is.na(bs_vector))) {
    bootstrap_ibs[b] <- calc_ibs(bs_vector, time_grid)
    cat(sprintf("Bootstrap %d: IBS = %.4f\n", b, bootstrap_ibs[b]))
  } else {
    bootstrap_ibs[b] <- NA
    cat(sprintf("Bootstrap %d: Failed to calculate IBS\n", b))
  }
}

# ==============================================================================
# 4. REPORTING RESULTS
# ==============================================================================

clean_ibs <- bootstrap_ibs[!is.na(bootstrap_ibs)]

if (length(clean_ibs) > 0) {
  mean_ibs <- mean(clean_ibs)
  se_ibs <- sd(clean_ibs) / sqrt(length(clean_ibs))
  ci_ibs <- mean_ibs + c(-1, 1) * qt(0.975, df = length(clean_ibs) - 1) * se_ibs
  
  cat("\n--- RESULTS: Integrated Brier Score (Calibration) ---\n")
  cat(sprintf("Mean IBS: %.4f (Lower is better)\n", mean_ibs))
  cat(sprintf("95%% CI: %.4f – %.4f\n", ci_ibs[1], ci_ibs[2]))
  
  cat("\nInterpretation Guide:\n")
  cat("  < 0.10: Excellent calibration\n")
  cat("  ~ 0.25: Random guessing (reference point)\n")
} else {
  cat("\nError: All IBS calculations failed.\n")
}

# After you compute: time_grid, cindex_list
plot(time_grid, cindex_list, type = "l",
     xlab = "Time", ylab = "Time-dependent C-index",
     main = "AF: C-index over time (Random Survival Forest)")

abline(h = 0.5, lty = 2)  # random baseline
points(time_grid, cindex_list, pch = 16, cex = 0.4)

bs_test <- calc_cr_brier(
  times = test_data$year,
  status = test_data$status,
  pred_cif = cif_matrix,
  time_grid = time_grid,
  cause_of_interest = 1
)

# =========================================================
# 6. Compute and save mean CIF over all test subjects
# =========================================================
mean_cif <- colMeans(cif_matrix, na.rm = TRUE)

rsf_mean_cif_df <- data.frame(
  time = time_grid,
  mean_cif = mean_cif
)

write.csv(rsf_mean_cif_df,
          "/home/UT_shared/result/rsf_mean_cif_cause1.csv",
          row.names = FALSE)
cat("Saved RSF mean CIF to /home/UT_shared/result/rsf_mean_cif_cause1.csv\n")

plot(time_grid, bs_test, type = "l",
     xlab = "Time", ylab = "Brier score",
     main = "AF: Brier score over time (Random Survival Forest)")
points(time_grid, bs_test, pch = 16, cex = 0.4)

rsf_df <- data.frame(time = time_grid, cindex = cindex_list)
write.csv(rsf_df, "/home/UT_shared/result/rsf_cindex_curve.csv", row.names = FALSE)
cat("Saved RSF curve to /home/UT_shared/result/rsf_cindex_curve.csv\n")

rsf_bs = data.frame(time = time_grid, bs = bs_test)
write.csv(rsf_bs, "/home/UT_shared/result/rsf_brier_curve.csv", row.names = FALSE)
cat("Saved RSF curve to /home/UT_shared/result/rsf_brier_curve.csv\n")
