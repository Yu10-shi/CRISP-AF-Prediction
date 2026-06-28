library(shiny)
library(shinyjs)
library(DT)
library(reticulate)
use_condaenv("/Users/jasperzzz/miniconda3/envs/hu_ssc", required = TRUE)
library(jsonlite)
library(ggplot2)
library(gridExtra)
library(scales)

# Allow large CSV uploads (e.g. 100MB)
options(shiny.maxRequestSize = 100 * 1024^2)

# Ensure working directory is the app directory
if (file.exists("app.R")) {
  # already correct
} else {
  script_dir <- tryCatch(dirname(normalizePath(sys.frames()[[1]]$ofile)), error = function(e) NULL)
  if (!is.null(script_dir) && dir.exists(script_dir)) setwd(script_dir)
}

# -----------------------------
# Global setup
# -----------------------------

source_python("predict_model.py")
meta <- jsonlite::read_json("model_meta.json", simplifyVector = TRUE)
model_candidates <- c(
  "/home/UT_shared/result/500_embedding_duration_10_deephit_tabtransformer_shared_node_epoch50.pt",
  "end-lr0.0001-wd0.01-eta0.8-alpha0.2-bs256-drop0.1-epochs100-dur30-embed500.pt"
)
model_candidates <- normalizePath(model_candidates[file.exists(model_candidates)], mustWork = FALSE)
MODEL_PATH <- model_candidates[1]
if (is.na(MODEL_PATH) || MODEL_PATH == "") stop("No model file found; update MODEL_PATH.")

load_model(MODEL_PATH)

cat_features <- meta$categ_idx
cont_features <- meta$cont_idx
feature_names <- meta$feature_names
cat_names <- feature_names[seq_along(cat_features)]
cont_names <- feature_names[(length(cat_features) + 1):length(feature_names)]
duration_labels <- meta$duration_labels
risk_names <- meta$risk_names

bin_year_labels <- sapply(0:9, function(i) {
  yr <- round(i / 9 * 14.9, 1)
  paste0("bin_", i, "\n(", yr, " yr)")
})

`%||%` <- function(a, b) if (!is.null(a) && length(a) > 0 && !anyNA(a)) a else b

# Shortened label for display; full name in tooltip
truncate_label <- function(x, max_len = 28) {
  ifelse(nchar(x) > max_len, paste0(substr(x, 1, max_len - 3), "..."), x)
}

# Top 20 important features shown in Manual entry mode
TOP_FEATURES <- c(
  "sex_imp", "demographics_age_index_ecg", "alkaline_phophatase_peri",
  "hgb_peri", "cancer_any_icd10", "dyslipidemia_combined",
  "ecg_resting_qtc", "creatinine_peri", "ecg_resting_hr",
  "acei_arb_entresto", "copd_icd10", "beta_blocker_any_peri",
  "ecg_resting_qrs", "ecg_resting_pr", "plt_peri",
  "chloride_peri", "hypertension_icd10", "ccb_dihydro_peri",
  "diuretic_loop_peri", "sodium_peri"
)

# Fixed defaults for the 58 Non_Top_Features (population means / most-common values)
NON_TOP_DEFAULTS <- list(
  # continuous — population means from Participant Characteristics
  hct_peri                        = 0.42,
  rdw_peri                        = 13.48,
  wbc_peri                        = 8.62,
  inr_peri                        = 1.07,
  ptt_peri                        = 31.70,
  esr_peri                        = 17.44,
  crp_high_sensitive_peri         = 9.43,
  albumin_peri                    = 36.93,
  alanine_transaminase_peri       = 32.86,
  aspartate_transaminase_peri     = 47.09,
  bilirubin_total_peri            = 10.79,
  bilirubin_direct_peri           = 6.69,
  urea_peri                       = 5.64,
  urine_alb_cr_ratio_peri         = 12.14,
  potassium_peri                  = 4.04,
  ck_peri                         = 304.34,
  troponin_t_hs_peri_highest      = 231.55,
  glucose_fasting_peri_highest    = 6.05,
  glucose_random_peri_highest     = 7.18,
  hga1c_peri_highest              = 6.14,
  tchol_peri_highest              = 5.06,
  ldl_peri_highest                = 2.95,
  hdl_peri_lowest                 = 1.28,
  tg_peri_highest                 = 1.90,
  iron_peri                       = 15.26,
  tibc_peri                       = 56.88,
  ferritin_peri                   = 194.40,
  tsh_peri                        = 2.42,
  # categorical — all 0 (most common value)
  diabetes_combined                              = 0L,
  dcm_icd10                                     = 0L,
  hcm_icd10                                     = 0L,
  myocarditis_icd10_prior                       = 0L,
  pericarditis_icd10_prior                      = 0L,
  aortic_aneurysm_icd10                         = 0L,
  aortic_dissection_icd10_prior                 = 0L,
  pulmonary_htn_icd10                           = 0L,
  amyloid_icd10                                 = 0L,
  obstructive._sleep_apnea_icd10                = 0L,
  hyperthyroid_icd10                            = 0L,
  hypothyroid_icd10                             = 0L,
  rheumatoid_arthritis_icd10                    = 0L,
  sle_icd10                                     = 0L,
  sarcoid_icd10                                 = 0L,
  event_cv_hf_admission_icd10_prior             = 0L,
  event_cv_ep_vt_any_icd10_prior                = 0L,
  event_cv_ep_sca_survived_icd10_cci_prior      = 0L,
  event_cv_cns_stroke_ischemic_icd10_prior      = 0L,
  event_cv_cns_stroke_hemorrh_icd10_prior       = 0L,
  event_cv_cns_tia_icd10_prior                  = 0L,
  pci_prior                                     = 0L,
  cabg_prior                                    = 0L,
  transplant_heart_cci_prior                    = 0L,
  lvad_cci_prior                                = 0L,
  pacemaker_permanent_cci_prior                 = 0L,
  crt_cci_prior                                 = 0L,
  icd_cci_prior                                 = 0L,
  ecg_resting_paced                             = 0L,
  ecg_resting_bigeminy                          = 0L,
  ecg_resting_LBBB                              = 0L,
  ecg_resting_RBBB                              = 0L,
  ecg_resting_incomplete_LBBB                   = 0L,
  ecg_resting_incomplete_RBBB                   = 0L,
  ecg_resting_LAFB                              = 0L,
  ecg_resting_LPFB                              = 0L,
  ecg_resting_bifascicular_block                = 0L,
  ecg_resting_trifascicular_block               = 0L,
  ecg_resting_intraventricular_conduction_delay = 0L,
  anti_platelet_oral_non_asa_any_peri           = 0L,
  anti_coagulant_oral_any_peri                  = 0L,
  nitrates_any_peri                             = 0L,
  ivabradine_peri                               = 0L,
  ccb_non_dihydro_peri                          = 0L,
  diuretic_thiazide_peri                        = 0L,
  diuretic_low_ceiling_non_thiazide_peri        = 0L,
  diuretic_mra_peri                             = 0L,
  anti_arrhythmic_any_peri                      = 0L,
  digoxin_peri                                  = 0L,
  amyloid_therapeutics_diflunisal_peri          = 0L,
  smoking_cessation_oral_peri                   = 0L,
  acute_mi_angina_other                         = 0L
)

example_path <- "../df_imputed_synthetic.csv"
if (file.exists(example_path)) {
  tmp <- read.csv(example_path, nrows = 5, check.names = FALSE)
  if (all(feature_names %in% names(tmp))) {
    example_defaults <- tmp[1, feature_names]
  } else {
    example_defaults <- setNames(as.list(rep(0, length(feature_names))), feature_names)
  }
} else {
  example_defaults <- setNames(as.list(rep(0, length(feature_names))), feature_names)
}

# -----------------------------
# UI
# -----------------------------

ui <- navbarPage(
  title = "AF & All-Cause Mortality Risk Prediction",
  id = "main_nav",
  header = useShinyjs(),
  tabPanel(
    "Predict",
    fluidPage(
      fluidRow(
        # ---------- Left panel: Data Entry ----------
        column(
          width = 4,
          wellPanel(
            style = "background: #f8f9fa; border: 1px solid #dee2e6;",
            h4("Data Entry", style = "margin-top:0;font-weight:bold"),
            radioButtons(
              "input_mode", "Input method:",
              choices = c("Manual entry", "Upload CSV"),
              inline = TRUE
            ),
            conditionalPanel(
              condition = "input.input_mode == 'Upload CSV'",
              tags$hr(),
              fileInput("csv_file", "CSV with required columns", accept = ".csv"),
              fluidRow(
                column(4, numericInput("cursor_index", "Current row:", value = 1, min = 1, max = 1, step = 1)),
                column(4, actionButton("cursor_up", "▲", title = "Previous row", style = "margin-top: 25px;")),
                column(4, actionButton("cursor_down", "▼", title = "Next row", style = "margin-top: 25px;"))
              ),
              helpText("Select a row to view its features and run prediction for that individual. Use arrows or type a number."),
              tags$hr()
            ),
            conditionalPanel(
              condition = "input.input_mode == 'Manual entry'",
              tags$hr()
            ),
            h5("Features", style = "margin-top:8px;font-weight:bold"),
            helpText("Label truncated; hover for full name. Integer codes for categorical.", style = "font-size:0.85em;"),
            div(
              style = "overflow-x: hidden;",
              uiOutput("manual_inputs_ui")
            ),
            conditionalPanel(
              condition = "input.input_mode == 'Manual entry'",
              actionButton("load_example", "Load example values", class = "btn btn-default btn-sm", style = "margin-top:8px;")
            )
          )
        ),
        # ---------- Middle panel: Prediction Output ----------
        column(
          width = 4,
          wellPanel(
            style = "background: #f8f9fa; border: 1px solid #dee2e6;",
            h4("Prediction Output", style = "margin-top:0;font-weight:bold"),
            selectInput("risk_head", "Outcome:", choices = risk_names, selected = "AF"),
            radioButtons("xaxis_mode", "X-axis:", choices = c("Bins" = "bin", "Years" = "year"), selected = "bin", inline = TRUE),
            actionButton("predict", "Run Prediction", class = "btn-primary btn-block", style = "margin-bottom:12px;"),
            fluidRow(
              column(6, downloadButton("download_plot_png", "Save curves (PNG)", style = "width: 100%; padding: 8px 12px; font-size: 14px;")),
              column(6, downloadButton("download_table_csv", "Save table (CSV)", style = "width: 100%; padding: 8px 12px; font-size: 14px;"))
            ),
            tags$div(style = "margin-bottom: 10px;"),

            # CP upper bound box — shown for whichever outcome is selected
            uiOutput("cp_upper_bound_ui"),

            h5("Event probability", style = "font-weight:bold; margin-top:8px;"),
            plotOutput("predictionPlot_prob", height = "200px"),
            h5("Survival", style = "font-weight:bold; margin-top:8px;"),
            plotOutput("predictionPlot_surv", height = "200px"),
            DTOutput("predictionTable")
          )
        ),
        # ---------- Right panel: Feature Importance ----------
        column(
          width = 4,
          wellPanel(
            style = "background: #f8f9fa; border: 1px solid #dee2e6;",
            h4("Feature Importance", style = "margin-top:0;font-weight:bold"),
            helpText("Test-case level: which features most influence this individual's prediction.", style = "font-size:0.85em;"),
            uiOutput("importance_scope_note"),
            selectInput("importance_time_bin", "Importance metric:", 
              choices = setNames(
                as.character(c(-2, -1, seq_along(duration_labels) - 1L)), 
                c("Overall (cumulative risk)", "Last bin", if (length(duration_labels) > 0) duration_labels else paste0("bin_", seq_len(10) - 1))
              ), selected = "-2"),
            actionButton("compute_importance", "Compute Importance", class = "btn-primary btn-block", style = "margin-bottom:10px;"),
            downloadButton("download_importance_csv", "Save importance (CSV)", style = "width: 100%; padding: 8px 12px; font-size: 14px; margin-bottom:10px;"),
            h5("Top features (sensitivity)", style = "font-weight:bold; margin-top:10px;"),
            plotOutput("importancePlot", height = "380px"),
            DTOutput("importanceTable")
          )
        )
      )
    )
  ),
  tabPanel(
    "Help",
    fluidPage(
      wellPanel(
        h4("AF & All-Cause Mortality Risk Prediction - Help", style = "margin-top:0;font-weight:bold"),
        p("This application estimates short-term risk patterns for two outcomes: AF (atrial fibrillation) and All cause death. It uses a TabTransformer + DeepHit model and returns per-bin event probability and derived survival curves for the selected outcome."),
        tags$hr(),
        h5("Model description", style = "font-weight:bold"),
        tags$ul(
          tags$li("Inputs: 78 features in fixed model order (61 categorical + 17 continuous)."),
          tags$li("Architecture: TabTransformer encoder with DeepHit-style risk-specific output heads."),
          tags$li("Time output: 10 discrete bins. For each outcome, the model outputs event probability by bin."),
          tags$li("Survival is computed as 1 minus the cumulative event probability across bins.")
        ),
        tags$hr(),
        h5("How to use this 3-panel interface", style = "font-weight:bold"),
        h6("Panel 1: Data Entry", style = "font-weight:bold"),
        tags$ul(
          tags$li("Choose input mode: Manual entry or Upload CSV."),
          tags$li("Manual entry: only the top 20 most important features are shown as inputs. The remaining 58 features are automatically set to their population mean values (continuous) or most common values (categorical) from the study cohort — this simplifies data entry while still providing a complete 78-feature vector to the model. Labels are shortened and show full names on hover. Use Load example values to populate fields for testing."),
          tags$li("Upload CSV: upload a CSV with required columns (one row per individual). Select Current row to view that row's features; Run Prediction uses exactly that selected row.")
        ),
        h6("Panel 2: Prediction Output", style = "font-weight:bold"),
        tags$ul(
          tags$li("Choose outcome: AF (default) or All cause death."),
          tags$li("Click Run Prediction to generate outputs."),
          tags$li("Both event probability and survival are shown: two plots and one table with columns Duration, Event probability, and Survival."),
          tags$li("In CSV mode, Run Prediction predicts for the selected row only; change the row and click Run Prediction again for another individual.")
        ),
        h6("Panel 3: Feature Importance", style = "font-weight:bold"),
        tags$ul(
          tags$li("Test-case level: shows which features most influence this individual's prediction."),
          tags$li("Uses the same patient data as the last Run Prediction (current row or manual entry)."),
          tags$li("Importance metric: Overall (cumulative risk, default), Last bin, or a specific bin."),
          tags$li(tags$strong("Compute Importance does NOT run automatically after Run Prediction."), " You must click the Compute Importance button manually each time you want importance results."),
          tags$li(tags$strong("Importance is outcome-specific."), " To get importance for AF: select AF in the Outcome selector (Panel 2) and click Compute Importance. To get importance for All cause death: switch the Outcome selector to All cause death and click Compute Importance again. Each outcome produces different importance scores."),
          tags$li("Importance resets automatically after each Run Prediction — you must click Compute Importance again to see updated results."),
          tags$li("Requires reference data: CSV mode with 2+ rows (other rows used as reference), or example_input.csv for manual mode."),
          tags$li("Save importance (CSV): downloads Feature and Importance columns.")
        ),
        tags$hr(),
        h5("Save results", style = "font-weight:bold"),
        tags$ul(
          tags$li("Save curves (PNG): downloads both event probability and survival curves in one image."),
          tags$li("Save table (CSV): downloads the table with Duration, Event probability, and Survival columns."),
          tags$li("File names include the selected outcome.")
        ),
        tags$hr(),
        h5("Notes", style = "font-weight:bold"),
        tags$ul(
          tags$li("CSV upload limit is set to 100 MB."),
          tags$li("Categorical values should use integer codes aligned with the model training schema."),
          tags$li("In CSV mode, if Current row is out of range (e.g., greater than the number of rows in the file), the app shows a warning and adjusts to a valid row."),
          tags$li("Feature importance uses permutation (sensitivity): each feature is replaced with a random value from the reference; importance = |change in prediction|."),
          tags$li("In Manual entry mode, importance is computed for the top 20 features only; the remaining 58 features are held at their population mean/default values. In CSV mode, importance is computed for all 78 features from the uploaded patient record.")
        )
      )
    )
  )
)

# -----------------------------
# Server
# -----------------------------

server <- function(input, output, session) {

  # Data Entry: Top 20 features only in Manual mode
  output$manual_inputs_ui <- renderUI({
    top_cat <- TOP_FEATURES[TOP_FEATURES %in% cat_names]
    top_cont <- TOP_FEATURES[TOP_FEATURES %in% cont_names]
    tagList(
      lapply(top_cat, function(feat) {
        fluidRow(
          column(
            width = 6,
            tags$label(
              `for` = paste0("feat_", feat),
              title = feat,
              style = "font-size: 0.9em; padding-top: 6px;",
              truncate_label(feat)
            ),
          ),
          column(
            width = 6,
            numericInput(
              inputId = paste0("feat_", feat),
              label = NULL,
              value = as.integer(example_defaults[[feat]] %||% 0),
              min = 0,
              step = 1
            )
          )
        )
      }),
      tags$hr(style = "margin: 8px 0;"),
      lapply(top_cont, function(feat) {
        fluidRow(
          column(
            width = 6,
            tags$label(
              `for` = paste0("feat_", feat),
              title = feat,
              style = "font-size: 0.9em; padding-top: 6px;",
              truncate_label(feat)
            ),
          ),
          column(
            width = 6,
            numericInput(
              inputId = paste0("feat_", feat),
              label = NULL,
              value = as.numeric(example_defaults[[feat]] %||% 0),
              step = 0.1
            )
          )
        )
      })
    )
  })

  manualData <- reactive({
    # Read the 20 top feature inputs
    top_vals <- setNames(
      lapply(TOP_FEATURES, function(feat) {
        v <- input[[paste0("feat_", feat)]] %||% 0
        if (feat %in% cat_names) as.integer(v) else as.numeric(v)
      }),
      TOP_FEATURES
    )
    # Merge with non-top defaults
    all_vals <- c(top_vals, NON_TOP_DEFAULTS)
    # Build data.frame in feature_names order
    df <- as.data.frame(
      lapply(feature_names, function(f) all_vals[[f]]),
      stringsAsFactors = FALSE,
      check.names = FALSE
    )
    names(df) <- feature_names
    df
  })

  # One-time CSV storage: parsed when file is uploaded, cleared on mode switch
  csv_store <- reactiveVal(NULL)

  csv_data <- reactive({
    req(input$csv_file)
    df <- tryCatch(
      read.csv(input$csv_file$datapath, check.names = FALSE, stringsAsFactors = FALSE),
      error = function(e) NULL
    )
    shiny::validate(need(!is.null(df), "Failed to read CSV"))
    missing_cols <- setdiff(feature_names, names(df))
    shiny::validate(
      need(length(missing_cols) == 0,
           paste("Missing required columns:", paste(missing_cols, collapse = ", ")))
    )
    csv_store(df)  # store for prediction (one-time read)
    df
  })

  # Cursor index, validated against CSV row count
  cursor_index_valid <- reactive({
    if (input$input_mode != "Upload CSV") return(1L)
    req(input$csv_file)
    df <- csv_data()
    n <- nrow(df)
    cur <- suppressWarnings(as.integer(input$cursor_index))
    if (is.na(cur) || cur < 1) return(1L)
    if (cur > n) return(NULL)  # out of range
    cur
  })

  # On mode switch: refresh everything, clear results, reset state
  observeEvent(input$input_mode, {
    rv$result <- NULL
    rv$cp_result <- NULL
    rv$nrows <- 0
    rv$importance <- NULL
    csv_store(NULL)
    if (input$input_mode == "Manual entry") {
      for (feat in TOP_FEATURES) {
        val <- example_defaults[[feat]] %||% 0
        if (feat %in% cat_names) {
          updateNumericInput(session, paste0("feat_", feat), value = as.integer(val))
        } else {
          updateNumericInput(session, paste0("feat_", feat), value = as.numeric(val))
        }
      }
    } else {
      updateNumericInput(session, "cursor_index", value = 1, min = 1, max = 1)
      # Sync features when switching to CSV (if file already loaded)
      tryCatch({
        if (!is.null(input$csv_file)) sync_csv_to_inputs()
      }, error = function(e) NULL)
    }
  })

  # Update cursor_index max and sync features when CSV loads
  observeEvent(csv_data(), {
    if (input$input_mode != "Upload CSV") return()
    df <- csv_data()
    n <- nrow(df)
    cur <- suppressWarnings(as.integer(input$cursor_index %||% 1))
    if (is.na(cur) || cur < 1) cur <- 1L
    val <- min(max(1L, cur), n)
    updateNumericInput(session, "cursor_index", max = n, min = 1, value = val)
    sync_csv_to_inputs()  # Refresh feature display for current row
  }, ignoreInit = TRUE)

  # cursor_row(): single 1-row data.frame for currently selected row (CSV mode) or manual inputs
  cursor_row <- reactive({
    if (input$input_mode == "Upload CSV") {
      req(input$csv_file)
      idx <- cursor_index_valid()
      if (is.null(idx)) return(NULL)  # out of range
      df <- csv_data()
      df <- df[, feature_names, drop = FALSE]
      row <- df[idx, , drop = FALSE]
    } else {
      row <- manualData()
      names(row) <- feature_names
    }
    row[, cat_names] <- lapply(row[, cat_names, drop = FALSE], function(x) as.integer(as.character(x) %||% 0))
    row[, cont_names] <- lapply(row[, cont_names, drop = FALSE], function(x) as.numeric(as.character(x) %||% 0))
    row
  })

  # Sync feature entries when CSV loads or cursor changes
  sync_csv_to_inputs <- function() {
    if (input$input_mode != "Upload CSV") return()
    if (is.null(input$csv_file)) return()
    df <- tryCatch(csv_data(), error = function(e) NULL)
    if (is.null(df) || nrow(df) == 0) return()
    n <- nrow(df)
    cur <- suppressWarnings(as.integer(input$cursor_index %||% 1))
    if (is.na(cur) || cur < 1 || cur > n) {
      if (!is.na(cur) && (cur < 1 || cur > n)) {
        showNotification(paste0("Current row must be between 1 and ", n, ". Adjusting."), type = "warning", duration = 4)
        updateNumericInput(session, "cursor_index", value = max(1L, min(cur, n)), max = n)
      }
      return()
    }
    row <- df[cur, feature_names, drop = FALSE]
    for (feat in cat_names) {
      val <- row[[feat]]
      if (is.factor(val)) val <- as.character(val)
      updateNumericInput(session, paste0("cat_", feat), value = as.integer(suppressWarnings(as.numeric(val)) %||% 0))
    }
    for (feat in cont_names) {
      val <- row[[feat]]
      if (is.factor(val)) val <- as.character(val)
      updateNumericInput(session, paste0("cont_", feat), value = as.numeric(suppressWarnings(as.numeric(val)) %||% 0))
    }
  }

  observeEvent(csv_data(), sync_csv_to_inputs(), ignoreInit = TRUE)
  observeEvent(input$cursor_index, sync_csv_to_inputs(), ignoreInit = TRUE)

  observeEvent(input$load_example, {
    for (feat in TOP_FEATURES) {
      val <- example_defaults[[feat]] %||% 0
      if (feat %in% cat_names) {
        updateNumericInput(session, paste0("feat_", feat), value = as.integer(val))
      } else {
        updateNumericInput(session, paste0("feat_", feat), value = as.numeric(val))
      }
    }
    # Loading example values = new data entry: stale both panels
    rv$result <- NULL
    rv$cp_result <- NULL
    rv$importance <- NULL
  })

  # Up/down buttons for row selection (CSV mode)
  observeEvent(input$cursor_up, {
    if (input$input_mode != "Upload CSV") return()
    cur <- suppressWarnings(as.integer(input$cursor_index %||% 1))
    if (is.na(cur) || cur <= 1) return()
    updateNumericInput(session, "cursor_index", value = cur - 1L)
  })
  observeEvent(input$cursor_down, {
    if (input$input_mode != "Upload CSV") return()
    df <- tryCatch(csv_data(), error = function(e) NULL)
    if (is.null(df) || nrow(df) == 0) return()
    cur <- suppressWarnings(as.integer(input$cursor_index %||% 1))
    if (is.na(cur) || cur >= nrow(df)) return()
    updateNumericInput(session, "cursor_index", value = cur + 1L)
  })

  # CSV mode: cursor change → stale both panels
  observeEvent(input$cursor_index, {
    if (input$input_mode == "Upload CSV") {
      rv$result <- NULL
      rv$cp_result <- NULL
      rv$importance <- NULL
    }
  }, ignoreInit = TRUE)

  # Manual mode: any feature input change → stale both panels
  observeEvent(manualData(), {
    if (input$input_mode == "Manual entry") {
      rv$result <- NULL
      rv$cp_result <- NULL
      rv$importance <- NULL
    }
  }, ignoreInit = TRUE)

  # Input df for prediction: single row (selected row in CSV mode, manual in Manual mode)
  input_df <- reactive({
    if (input$input_mode == "Upload CSV") {
      row <- cursor_row()  # exactly the selected row
      if (is.null(row) || nrow(row) == 0) return(NULL)
      df <- row
    } else {
      df <- manualData()
      names(df) <- feature_names
    }
    # Robust conversion: handle factor, character, NA
    for (f in cat_names) {
      v <- df[[f]]
      if (is.factor(v)) v <- as.character(v)
      df[[f]] <- as.integer(suppressWarnings(as.numeric(v)) %||% 0L)
    }
    for (f in cont_names) {
      v <- df[[f]]
      if (is.factor(v)) v <- as.character(v)
      df[[f]] <- as.numeric(suppressWarnings(as.numeric(v)) %||% 0)
    }
    df
  })

  rv <- reactiveValues(result = NULL, nrows = 0, importance = NULL, importance_version = 0L, cp_result = NULL)

  # ── CP upper bound: select outcome (AF / death) and current patient row ─────
  cp_current <- reactive({
    cp <- rv$cp_result
    if (is.null(cp)) return(NULL)
    ev  <- if (identical(input$risk_head, "AF")) cp$af else cp$death
    if (is.null(ev)) return(NULL)
    row <- tryCatch(display_row_idx(), error = function(e) 1L)
    n   <- length(ev$upper_bound_year)
    row <- max(1L, min(as.integer(row), n))
    list(
      ub_year   = round(ev$upper_bound_year[[row]], 2),
      ub_bin    = ev$upper_bound_bin[[row]],
      conf      = round(cp$target_coverage * 100),
      gamma     = cp$gamma,
      label     = if (identical(input$risk_head, "AF")) "AF" else "All-cause death"
    )
  })

  # ── CP upper bound UI renderer ─────────────────────────────────────────────
  output$cp_upper_bound_ui <- renderUI({
    cur <- cp_current()
    if (is.null(cur)) {
      # placeholder before first prediction
      div(
        style = "background:#f0f4ff; border-left:4px solid #4a90d9;
                 padding:10px 14px; border-radius:4px; margin-bottom:10px;",
        tags$small(style = "color:#666;",
          icon("info-circle"), " Run Prediction to see the conformal upper bound."
        )
      )
    } else {
      div(
        style = "background:#e8f5e9; border-left:4px solid #2e7d32;
                 padding:10px 14px; border-radius:4px; margin-bottom:10px;",
        tags$strong(style = "color:#1b5e20; font-size:1.05em;",
          icon("shield-alt"),
          sprintf(" %s Upper Bound: %.2f years", cur$label, cur$ub_year)
        ),
        tags$br(),
        tags$small(style = "color:#2e7d32;",
          sprintf("With %d%% confidence, %s is expected within %.2f years",
                  cur$conf, cur$label, cur$ub_year)
        ),
        tags$br(),
        tags$small(style = "color:#888; font-size:0.8em;",
          sprintf("(gamma = %.2f | AJ-recalibrated UPB)", cur$gamma)
        )
      )
    }
  })
  # ──────────────────────────────────────────────────────────────────────────

  # Reference data for permutation importance: sample replacement values from here
  reference_df <- reactive({
    if (input$input_mode == "Upload CSV" && !is.null(input$csv_file)) {
      df <- tryCatch(csv_data(), error = function(e) NULL)
      if (is.null(df) || nrow(df) == 0) return(NULL)
      idx <- cursor_index_valid()
      if (is.null(idx)) return(NULL)
      # Exclude current row so we sample from other individuals
      if (nrow(df) > 1) {
        ref <- df[-idx, feature_names, drop = FALSE]
      } else {
        ref <- NULL  # single row: need external reference
      }
    } else {
      ref <- NULL
    }
    # Manual mode or single-row CSV: try multiple paths for reference
    if (is.null(ref) || nrow(ref) == 0) {
      ref_paths <- c(
        "example_input.csv",
        file.path(getwd(), "example_input.csv"),
        example_path,
        file.path(getwd(), example_path)
      )
      for (path in unique(ref_paths)) {
        if (!is.null(path) && nzchar(path) && file.exists(path)) {
          tmp <- tryCatch(read.csv(path, check.names = FALSE, stringsAsFactors = FALSE), error = function(e) NULL)
          if (!is.null(tmp) && nrow(tmp) >= 1 && all(feature_names %in% names(tmp))) {
            ref <- tmp[, feature_names, drop = FALSE]
            rownames(ref) <- NULL
            break
          }
        }
      }
    }
    ref
  })

  observeEvent(input$predict, {
    rv$result <- NULL  # Clear previous result first to avoid stale/crash state
    rv$cp_result <- NULL
    err_msg <- NULL
    if (input$input_mode == "Upload CSV" && is.null(input$csv_file)) {
      showNotification("Please upload a CSV file first.", type = "error", duration = 5)
      return()
    }
    df <- tryCatch(
      input_df(),
      error = function(e) {
        err_msg <<- e$message
        NULL
      }
    )
    if (is.null(df) || nrow(df) < 1) {
      showNotification(
        if (nzchar(err_msg %||% "")) paste("Error loading data:", err_msg)
        else if (input$input_mode == "Upload CSV") "No row selected or row out of range. Check Current row."
        else "No rows to predict. Check your input.",
        type = "error", duration = 8
      )
      return()
    }
    x_cat <- tryCatch(as.matrix(df[, cat_names, drop = FALSE]), error = function(e) NULL)
    x_cont <- tryCatch(as.matrix(df[, cont_names, drop = FALSE]), error = function(e) NULL)
    if (is.null(x_cat) || is.null(x_cont)) {
      showNotification("Error preparing data for prediction.", type = "error", duration = 5)
      return()
    }
    if (any(is.na(x_cat)) || any(is.na(x_cont))) {
      showNotification("Data contains NA values. Please check your CSV or manual entries.", type = "error", duration = 5)
      return()
    }
    meta_path_abs <- tryCatch(normalizePath("model_meta.json", mustWork = TRUE), error = function(e) NULL)
    if (is.null(meta_path_abs)) {
      showNotification("model_meta.json not found.", type = "error", duration = 5)
      return()
    }

    row_info <- if (input$input_mode == "Upload CSV") paste("row", input$cursor_index %||% 1) else "1 row"
    showNotification(paste("Running prediction for", row_info, "..."), type = "message", duration = 2)
    res <- tryCatch(
      {
        load_model(MODEL_PATH)  # Ensure model is loaded before each prediction
        message("Calling Python predict... rows: ", nrow(df))
        predict_from_arrays(
          x_cat = x_cat,
          x_cont = x_cont,
          model_path = MODEL_PATH,
          meta_path = meta_path_abs
        )
      },
      error = function(e) e
    )
    if (inherits(res, "error")) {
      rv$result <- NULL
      rv$cp_result <- NULL
      showNotification(paste("Prediction error:", res$message), type = "error", duration = 8)
      return()
    }
    rv$result <- res
    rv$nrows <- nrow(df)

    # ── CP upper bound (AF + all-cause death), AJ-recalibrated UPB ──────────
    recalib_path_abs <- normalizePath("cp_outputs/ucttrp_aj_upb_recalibration.npz", mustWork = FALSE)
    if (file.exists(recalib_path_abs)) {
      cp_res <- tryCatch(
        predict_event_upper_bounds(
          x_cat        = x_cat,
          x_cont       = x_cont,
          model_path   = MODEL_PATH,
          meta_path    = meta_path_abs,
          recalib_path = recalib_path_abs
        ),
        error = function(e) { message("CP error: ", e$message); NULL }
      )
      rv$cp_result <- cp_res
    } else {
      rv$cp_result <- NULL
    }
    # ──────────────────────────────────────────────────────────────────────

    showNotification("Prediction complete", type = "message", duration = 3)
  })

  # Sanitize data.frame for Python (ensure numeric types, no factors)
  sanitize_for_python <- function(df) {
    if (is.null(df) || nrow(df) == 0) return(NULL)
    out <- df[, feature_names, drop = FALSE]
    rownames(out) <- NULL
    for (f in cat_names) out[[f]] <- as.integer(suppressWarnings(as.numeric(as.character(out[[f]]))) %||% 0L)
    for (f in cont_names) out[[f]] <- as.numeric(suppressWarnings(as.numeric(as.character(out[[f]]))) %||% 0)
    out
  }

  # Compute permutation importance for the current sample (same row sent to prediction)
  observeEvent(input$compute_importance, {
    rv$importance <- NULL
    test_row <- tryCatch(sanitize_for_python(cursor_row()), error = function(e) NULL)
    ref <- tryCatch(sanitize_for_python(reference_df()), error = function(e) NULL)
    if (is.null(test_row) || nrow(test_row) == 0) {
      showNotification("No sample to analyze. Enter features or select a row.", type = "error", duration = 5)
      return()
    }
    if (is.null(ref) || nrow(ref) < 1) {
      showNotification("No reference data for permutation. Use CSV mode (multi-row) or add example_input.csv.", type = "error", duration = 6)
      return()
    }
    meta_path_abs <- tryCatch(
      normalizePath("model_meta.json", mustWork = TRUE),
      error = function(e) {
        p <- file.path(getwd(), "model_meta.json")
        if (file.exists(p)) normalizePath(p) else NULL
      }
    )
    if (is.null(meta_path_abs)) {
      showNotification("model_meta.json not found. Run app from ShinyApp directory.", type = "error", duration = 5)
      return()
    }
    risk_idx <- match(input$risk_head, risk_names)
    if (is.na(risk_idx)) risk_idx <- 1L
    risk_idx <- risk_idx - 1L
    time_bin <- suppressWarnings(as.integer(input$importance_time_bin %||% "-2"))
    if (is.na(time_bin)) time_bin <- -2L

    # Determine features to permute based on input mode (14.4)
    if (input$input_mode == "Manual entry") {
      features_to_permute <- as.list(TOP_FEATURES)
      non_top_defaults_py <- NON_TOP_DEFAULTS
    } else {
      features_to_permute <- as.list(feature_names)  # all 78
      non_top_defaults_py <- list()
    }

    # Mode-aware notification message (14.5)
    n_passes <- if (input$input_mode == "Manual entry") 20L else 78L
    scope_msg <- if (input$input_mode == "Manual entry") "top features only" else "all features"

    showNotification(
      paste0("Computing feature importance (", n_passes, " passes, ", scope_msg, ")..."),
      type = "message", duration = 3
    )
    imp <- tryCatch(
      permutation_importance_from_dataframe(
        df_test_row = test_row,
        df_ref = ref,
        model_path = MODEL_PATH,
        meta_path = meta_path_abs,
        risk_idx = risk_idx,
        time_bin = time_bin,
        features_to_permute = features_to_permute,
        non_top_defaults = non_top_defaults_py
      ),
      error = function(e) e
    )
    if (inherits(imp, "error")) {
      showNotification(paste("Importance error:", imp$message), type = "error", duration = 10)
      return()
    }
    rv$importance <- imp
    rv$importance_version <- rv$importance_version + 1L
    showNotification("Feature importance complete", type = "message", duration = 3)
  })

  # Display row index: clamp to prediction result rows; depends on cursor so outputs refresh on cursor change
  display_row_idx <- reactive({
    req(rv$result)
    num_rows <- dim(rv$result$prob)[1]
    idx <- if (input$input_mode == "Upload CSV") {
      as.integer(input$cursor_index %||% 1)
    } else {
      1L
    }
    max(1, min(idx, num_rows))
  })

  # Current selected series/table values (both event prob and survival)
  current_prediction_df <- reactive({
    req(rv$result)
    res <- rv$result
    probs <- res$prob
    surv <- res$surv
    num_risks <- dim(probs)[2]
    num_bins <- dim(probs)[3]

    row_idx <- display_row_idx()
    risk_idx <- max(1, min(match(input$risk_head, res$risk_names), num_risks))
    duration <- if (length(res$duration_labels) == num_bins) res$duration_labels else seq_len(num_bins)

    prob_vals <- as.numeric(probs[row_idx, risk_idx, ])
    surv_vals <- as.numeric(surv[row_idx, risk_idx, ])
    list(
      data = data.frame(
        Duration = duration,
        Event_probability = prob_vals,
        Survival = surv_vals,
        stringsAsFactors = FALSE
      ),
      risk_name = res$risk_names[risk_idx],
      row_idx = row_idx
    )
  })

  # Helper: x-position (bin index) of the CP upper bound for the current plot
  cp_vline_x <- reactive({
    cur <- cp_current()
    if (is.null(cur)) return(NULL)
    list(
      x       = cur$ub_bin + 1L,   # R is 1-indexed; Python bin 0 → idx 1
      ub_year = cur$ub_year,
      conf    = cur$conf
    )
  })

  # Shared helper: build x-axis labels and title
  .xaxis <- function(n_bins, mode) {
    if (isTRUE(mode == "year")) {
      list(
        labels = sapply(seq_len(n_bins) - 1L, function(i) round(i / (n_bins - 1) * 15.0, 1)),
        title  = "Follow-up (years)"
      )
    } else {
      list(labels = as.character(seq_len(n_bins)), title = "Bin")
    }
  }

  prediction_plot_prob <- reactive({
    cur    <- current_prediction_df()
    df_plot <- cur$data
    df_plot$idx <- seq_len(nrow(df_plot))
    n_bins <- nrow(df_plot)
    xa     <- .xaxis(n_bins, input$xaxis_mode)
    vl     <- cp_vline_x()

    p <- ggplot(df_plot, aes(x = idx, y = Event_probability)) +
      geom_line(color = "steelblue") +
      geom_point(color = "steelblue") +
      scale_x_continuous(breaks = df_plot$idx, labels = xa$labels) +
      labs(title = paste(cur$risk_name, "- Event probability"),
           x = xa$title, y = "Event probability") +
      theme_minimal() +
      theme(axis.text.x = element_text(size = 7))

    if (!is.null(vl)) {
      p <- p +
        geom_vline(xintercept = vl$x, color = "red", linewidth = 0.9, linetype = "dashed") +
        annotate("text",
                 x     = vl$x + 0.25,
                 y     = max(df_plot$Event_probability) * 0.95,
                 label = sprintf("%d%% UB: %.2f yr", vl$conf, vl$ub_year),
                 color = "red", size = 3, hjust = 0)
    }
    p
  })

  prediction_plot_surv <- reactive({
    cur    <- current_prediction_df()
    df_plot <- cur$data
    df_plot$idx <- seq_len(nrow(df_plot))
    n_bins <- nrow(df_plot)
    xa     <- .xaxis(n_bins, input$xaxis_mode)
    vl     <- cp_vline_x()

    p <- ggplot(df_plot, aes(x = idx, y = Survival)) +
      geom_line(color = "steelblue") +
      geom_point(color = "steelblue") +
      scale_x_continuous(breaks = df_plot$idx, labels = xa$labels) +
      labs(title = paste(cur$risk_name, "- Survival"),
           x = xa$title, y = "Survival") +
      theme_minimal() +
      theme(axis.text.x = element_text(size = 7))

    if (!is.null(vl)) {
      p <- p +
        geom_vline(xintercept = vl$x, color = "red", linewidth = 0.9, linetype = "dashed") +
        annotate("text",
                 x     = vl$x + 0.25,
                 y     = 0.97,
                 label = sprintf("%d%% UB: %.2f yr", vl$conf, vl$ub_year),
                 color = "red", size = 3, hjust = 0)
    }
    p
  })

  output$predictionTable <- renderDT({
    if (is.null(rv$result)) {
      return(datatable(
        data.frame(Message = "Click 'Run Prediction' to see event probability and survival by bin."),
        rownames = FALSE,
        options = list(dom = "t", paging = FALSE, searching = FALSE, ordering = FALSE)
      ))
    }
    df_out <- current_prediction_df()$data
    df_out$Event_probability <- round(df_out$Event_probability, 4)
    df_out$Survival <- round(df_out$Survival, 4)
    datatable(df_out, rownames = FALSE, options = list(dom = "t", pageLength = 10))
  })

  output$predictionPlot_prob <- renderPlot({
    if (is.null(rv$result)) {
      par(mar = c(2, 2, 2, 2))
      plot.new()
      text(0.5, 0.5, "Click 'Run Prediction' to see plot.", cex = 1.2)
      return(invisible(NULL))
    }
    prediction_plot_prob()
  })

  output$predictionPlot_surv <- renderPlot({
    if (is.null(rv$result)) {
      par(mar = c(2, 2, 2, 2))
      plot.new()
      text(0.5, 0.5, "Click 'Run Prediction' to see plot.", cex = 1.2)
      return(invisible(NULL))
    }
    prediction_plot_surv()
  })

  output$download_table_csv <- downloadHandler(
    filename = function() {
      req(rv$result)
      cur <- current_prediction_df()
      risk <- gsub("[^A-Za-z0-9]+", "_", tolower(cur$risk_name))
      paste0("prediction_", risk, "_row_", cur$row_idx, ".csv")
    },
    content = function(file) {
      req(rv$result)
      df_out <- current_prediction_df()$data
      df_out$Event_probability <- round(df_out$Event_probability, 4)
      df_out$Survival <- round(df_out$Survival, 4)
      write.csv(df_out, file, row.names = FALSE)
    }
  )

  output$download_plot_png <- downloadHandler(
    filename = function() {
      req(rv$result)
      cur <- current_prediction_df()
      risk <- gsub("[^A-Za-z0-9]+", "_", tolower(cur$risk_name))
      paste0("prediction_curves_", risk, "_row_", cur$row_idx, ".png")
    },
    content = function(file) {
      req(rv$result)
      p_prob <- prediction_plot_prob()
      p_surv <- prediction_plot_surv()
      png(file, width = 8, height = 10, units = "in", res = 300)
      gridExtra::grid.arrange(p_prob, p_surv, ncol = 1)
      dev.off()
    }
  )

  # Clear importance when new prediction runs (user must re-click Compute Importance)
  observeEvent(rv$result, {
    rv$importance <- NULL
  }, ignoreNULL = FALSE, ignoreInit = TRUE)

  # Feature importance outputs
  importance_df <- reactive({
    rv$importance_version  # explicit dependency — forces invalidation on every new computation
    if (is.null(rv$importance)) return(NULL)
    imp <- rv$importance
    # Robustly convert Python dict / reticulate object to named R numeric vector
    if (!is.list(imp) && !is.numeric(imp)) {
      imp <- tryCatch(reticulate::py_to_r(imp), error = function(e) NULL)
    }
    if (is.null(imp)) return(NULL)
    nms <- names(imp)
    vals <- tryCatch(
      vapply(imp, function(x) {
        v <- suppressWarnings(as.numeric(x))
        if (length(v) == 0 || is.na(v[1])) 0 else v[1]
      }, numeric(1)),
      error = function(e) NULL
    )
    if (is.null(vals) || is.null(nms) || length(nms) != length(vals)) return(NULL)
    df <- data.frame(Feature = nms, Importance = vals, stringsAsFactors = FALSE)
    df <- df[order(-df$Importance, df$Feature), , drop = FALSE]
    df
  })

  # Mode-aware scope note for Feature Importance panel (14.6)
  output$importance_scope_note <- renderUI({
    if (input$input_mode == "Manual entry") {
      helpText("Scope: top 20 features; others held at population means.",
               style = "font-size:0.82em; color:#666; font-style:italic;")
    } else {
      helpText("Scope: all 78 features from the uploaded CSV.",
               style = "font-size:0.82em; color:#666; font-style:italic;")
    }
  })

  output$importancePlot <- renderPlot({    df <- importance_df()
    if (is.null(df) || nrow(df) == 0) {
      par(mar = c(2, 2, 2, 2))
      plot.new()
      text(0.5, 0.5, "Click 'Compute Importance' to see which features\nmost influence this individual's prediction.", cex = 0.95)
      return(invisible(NULL))
    }
    df_nonzero <- df[df$Importance > 0, , drop = FALSE]
    if (nrow(df_nonzero) == 0) {
      par(mar = c(2, 2, 2, 2))
      plot.new()
      text(0.5, 0.5, "All feature importances are zero.\nThe prediction may be insensitive to feature changes\nfor this patient with the current reference data.", cex = 0.9)
      return(invisible(NULL))
    }
    top_n <- min(20, nrow(df_nonzero))
    top <- head(df_nonzero, top_n)

    # Normalize to % of total sensitivity
    total <- sum(df_nonzero$Importance)
    top$Pct <- top$Importance / total * 100

    # Shared y-axis order (highest importance at top)
    feat_levels <- rev(top$Feature)
    top$Feature <- factor(top$Feature, levels = feat_levels)

    # Left plot: % of total
    p_pct <- ggplot(top, aes(x = Pct, y = Feature)) +
      geom_col(fill = "steelblue", width = 0.7) +
      geom_text(aes(label = sprintf("%.1f%%", Pct)), hjust = -0.1, size = 2.8) +
      scale_x_continuous(expand = expansion(mult = c(0, 0.18))) +
      labs(x = "% of total sensitivity", y = NULL,
           title = "Relative importance") +
      theme_minimal() +
      theme(axis.text.y = element_text(size = 8),
            axis.text.x = element_text(size = 7),
            plot.title = element_text(size = 9, face = "bold"))

    # Right plot: raw absolute change (scientific notation)
    p_raw <- ggplot(top, aes(x = Importance, y = Feature)) +
      geom_col(fill = "coral3", width = 0.7) +
      scale_x_continuous(labels = scales::scientific,
                         expand = expansion(mult = c(0, 0.15))) +
      labs(x = "|Δ prediction|", y = NULL,
           title = "Raw sensitivity") +
      theme_minimal() +
      theme(axis.text.y = element_blank(),
            axis.ticks.y = element_blank(),
            axis.text.x = element_text(size = 7),
            plot.title = element_text(size = 9, face = "bold"))

    gridExtra::grid.arrange(p_pct, p_raw, ncol = 2, widths = c(1.6, 1))
  })

  output$importanceTable <- renderDT({
    df <- importance_df()
    if (is.null(df) || nrow(df) == 0) {
      return(datatable(
        data.frame(Message = "Click 'Compute Importance' to see feature importance for the current sample."),
        rownames = FALSE,
        options = list(dom = "t", paging = FALSE, searching = FALSE, ordering = FALSE)
      ))
    }
    total <- sum(df$Importance[df$Importance > 0])
    df$Pct <- ifelse(total > 0, round(df$Importance / total * 100, 2), 0)
    df$Importance <- formatC(df$Importance, format = "e", digits = 3)
    names(df) <- c("Feature", "Raw |Δ prediction|", "% of total")
    datatable(df, rownames = FALSE, options = list(pageLength = 15, order = list(list(2, "desc"))))
  })

  output$download_importance_csv <- downloadHandler(
    filename = function() {
      risk <- gsub("[^A-Za-z0-9]+", "_", tolower(input$risk_head %||% "AF"))
      paste0("feature_importance_", risk, ".csv")
    },
    content = function(file) {
      df <- importance_df()
      if (is.null(df) || nrow(df) == 0) {
        write.csv(data.frame(Message = "No importance data. Compute first."), file, row.names = FALSE)
        return()
      }
      write.csv(df, file, row.names = FALSE)
    }
  )
}

shinyApp(ui = ui, server = server)
