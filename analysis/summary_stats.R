# summary_stats.R
# ────────────────────────────────────────────────────────────────────────────────
# ClinicalNER De-identification Pipeline — R Statistical Analysis
#
# Purpose : Compute descriptive and inferential statistics on the processed
#           clinical notes corpus and generate publication-quality visuals.
#
# Requirements:
#   install.packages(c("RSQLite", "DBI", "dplyr", "tidyr", "ggplot2",
#                       "scales", "ggthemes", "patchwork", "broom",
#                       "jsonlite", "knitr"))
#
# Usage (from project root):
#   Rscript analysis/summary_stats.R
#   # or from RStudio: open and knit/source
# ────────────────────────────────────────────────────────────────────────────────

suppressPackageStartupMessages({
  library(DBI)
  library(RSQLite)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(scales)
  library(patchwork)
  library(broom)
  library(jsonlite)
})

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ROOT  <- here::here()  # or set manually: "/path/to/ClinicalNER"
DB_PATH       <- file.path(PROJECT_ROOT, "data", "clinicalner.db")
BM_PATH       <- file.path(PROJECT_ROOT, "data", "benchmark_results.json")
OUT_DIR       <- file.path(PROJECT_ROOT, "analysis", "output")
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

ACCENT <- "#4C72B0"
WARM   <- "#DD8452"

cat("=== ClinicalNER R Summary Statistics ===\n")
cat(sprintf("DB path: %s\nOutput : %s\n\n", DB_PATH, OUT_DIR))

# ── 1. Load Data ──────────────────────────────────────────────────────────────

load_data <- function(db_path) {
  if (!file.exists(db_path)) {
    message("[WARN] Database not found — using synthetic demo data.")
    set.seed(42)
    n <- 500
    specialties <- c("Cardiology", "Orthopedic", "Neurology", "Pulmonology",
                     "Gastroenterology", "Radiology", "Emergency", "Psychiatry")
    notes <- tibble(
      id          = seq_len(n),
      specialty   = sample(specialties, n, replace = TRUE),
      note_length = as.integer(runif(n, 200, 3000)),
      phi_count   = as.integer(runif(n, 0, 20)),
      note_date   = seq.Date(as.Date("2022-01-01"), by = "6 hours", length.out = n)
    )
    processed <- tibble(
      note_id        = seq_len(n),
      entity_count   = notes$phi_count,
      processed_at   = notes$note_date,
      processing_time_ms = as.integer(rnorm(n, 900, 200))
    )
    return(list(notes = notes, processed = processed))
  }

  con <- dbConnect(SQLite(), db_path)
  on.exit(dbDisconnect(con))

  tables <- dbListTables(con)
  notes     <- if ("clinical_notes"  %in% tables) dbReadTable(con, "clinical_notes")  else tibble()
  processed <- if ("processed_notes" %in% tables) dbReadTable(con, "processed_notes") else tibble()
  list(notes = as_tibble(notes), processed = as_tibble(processed))
}

dat <- load_data(DB_PATH)
notes     <- dat$notes
processed <- dat$processed

cat(sprintf("Notes loaded     : %s\n", nrow(notes) |> format(big.mark = ",")))
cat(sprintf("Processed loaded : %s\n\n", nrow(processed) |> format(big.mark = ",")))

# ── 2. Derive Features ────────────────────────────────────────────────────────

if (!"note_length" %in% colnames(notes)) {
  text_col <- intersect(c("transcription", "note_text", "text"), colnames(notes))[1]
  if (!is.na(text_col)) notes <- notes |> mutate(note_length = nchar(.data[[text_col]]))
}

if (!"phi_count" %in% colnames(notes) && "entity_count" %in% colnames(processed)) {
  join_key <- intersect(c("id", "note_id"), colnames(notes))[1]
  if (!is.na(join_key)) {
    notes <- notes |>
      left_join(processed |> select(note_id, entity_count), by = c("id" = "note_id")) |>
      rename(phi_count = entity_count)
  }
}

notes <- notes |>
  mutate(
    phi_density = phi_count / (note_length / 100),
    note_length = as.numeric(note_length),
    phi_count   = as.numeric(phi_count)
  )

# ── 3. Descriptive Statistics ─────────────────────────────────────────────────

cat("=== DESCRIPTIVE STATISTICS ===\n")

desc_stats <- notes |>
  summarise(across(c(note_length, phi_count, phi_density),
                   list(mean   = ~mean(.x, na.rm = TRUE),
                        sd     = ~sd(.x, na.rm = TRUE),
                        median = ~median(.x, na.rm = TRUE),
                        q25    = ~quantile(.x, 0.25, na.rm = TRUE),
                        q75    = ~quantile(.x, 0.75, na.rm = TRUE),
                        min    = ~min(.x, na.rm = TRUE),
                        max    = ~max(.x, na.rm = TRUE)),
                   .names = "{.col}__{.fn}")) |>
  pivot_longer(everything(), names_to = c("variable", "stat"), names_sep = "__") |>
  pivot_wider(names_from = stat, values_from = value)

print(desc_stats, n = Inf)

# Save CSV
write.csv(desc_stats, file.path(OUT_DIR, "descriptive_stats.csv"), row.names = FALSE)
cat("\nSaved: descriptive_stats.csv\n\n")

# ── 4. Specialty Summary ──────────────────────────────────────────────────────

if ("specialty" %in% colnames(notes)) {
  spec_summary <- notes |>
    group_by(specialty) |>
    summarise(
      n            = n(),
      mean_length  = round(mean(note_length, na.rm = TRUE), 1),
      median_phi   = median(phi_count, na.rm = TRUE),
      mean_density = round(mean(phi_density, na.rm = TRUE), 3),
      .groups = "drop"
    ) |>
    arrange(desc(mean_density))

  cat("=== SPECIALTY SUMMARY ===\n")
  print(spec_summary, n = Inf)
  write.csv(spec_summary, file.path(OUT_DIR, "specialty_summary.csv"), row.names = FALSE)
  cat("\nSaved: specialty_summary.csv\n\n")
}

# ── 5. Inferential Tests ─────────────────────────────────────────────────────

cat("=== INFERENTIAL STATISTICS ===\n")

# Pearson correlation: note_length ~ phi_count
if (all(c("note_length", "phi_count") %in% colnames(notes))) {
  ct <- cor.test(notes$note_length, notes$phi_count, use = "complete.obs")
  cat(sprintf("  Pearson r (length vs PHI count): %.3f  [95%% CI: %.3f, %.3f]  p = %.4f\n",
              ct$estimate, ct$conf.int[1], ct$conf.int[2], ct$p.value))
}

# Kruskal-Wallis: phi_density across specialties
if ("specialty" %in% colnames(notes) && "phi_density" %in% colnames(notes)) {
  kw <- kruskal.test(phi_density ~ specialty, data = notes)
  cat(sprintf("  Kruskal-Wallis (density ~ specialty): chi2(%.0f) = %.2f  p = %.4f\n",
              kw$parameter, kw$statistic, kw$p.value))
}

cat("\n")

# ── 6. Plots ──────────────────────────────────────────────────────────────────

theme_clinicalner <- theme_bw(base_size = 12) +
  theme(
    strip.background = element_rect(fill = "#EFF3FF"),
    panel.grid.minor = element_blank(),
    plot.title       = element_text(face = "bold", size = 13)
  )

## 6a. Note-length distribution
p_length <- ggplot(notes, aes(x = note_length)) +
  geom_histogram(bins = 50, fill = ACCENT, colour = "white", alpha = 0.85) +
  geom_vline(aes(xintercept = median(note_length, na.rm = TRUE)),
             linetype = "dashed", colour = WARM, linewidth = 1) +
  scale_x_continuous(labels = comma) +
  labs(title = "Note Length Distribution",
       x = "Note Length (characters)", y = "Count") +
  theme_clinicalner

## 6b. PHI density by specialty
if ("specialty" %in% colnames(notes)) {
  top_specs <- spec_summary |> slice_head(n = 8) |> pull(specialty)
  p_density <- notes |>
    filter(specialty %in% top_specs) |>
    ggplot(aes(x = reorder(specialty, phi_density, median), y = phi_density, fill = specialty)) +
    geom_boxplot(outlier.size = 0.7, alpha = 0.8) +
    coord_flip() +
    scale_fill_viridis_d(option = "muted", guide = "none") +
    labs(title = "PHI Density by Specialty",
         x = "", y = "Entities per 100 Characters") +
    theme_clinicalner
} else {
  p_density <- ggplot() + theme_void() + labs(title = "Specialty data not available")
}

## 6c. Scatter: note length vs PHI count
p_scatter <- ggplot(notes |> sample_n(min(500, nrow(notes))),
                    aes(x = note_length, y = phi_count)) +
  geom_point(alpha = 0.3, size = 1.2, colour = ACCENT) +
  geom_smooth(method = "lm", colour = WARM, se = TRUE, linewidth = 1.1) +
  scale_x_continuous(labels = comma) +
  labs(title = "Note Length vs. PHI Count",
       x = "Note Length (chars)", y = "PHI Entity Count") +
  theme_clinicalner

## 6d. Benchmark accuracy (if available)
if (file.exists(BM_PATH)) {
  bm <- tryCatch(fromJSON(BM_PATH), error = function(e) NULL)
  if (!is.null(bm)) {
    acc_df <- tibble(
      metric = c("Precision", "Recall", "F1"),
      value  = c(bm$precision %||% NA, bm$recall %||% NA, bm$f1_score %||% bm$f1 %||% NA)
    ) |> drop_na()

    p_bm <- ggplot(acc_df, aes(x = metric, y = value, fill = metric)) +
      geom_col(width = 0.5, alpha = 0.85) +
      geom_text(aes(label = sprintf("%.3f", value)), vjust = -0.5, size = 4) +
      scale_y_continuous(limits = c(0, 1.05)) +
      scale_fill_manual(values = c(Precision = ACCENT, Recall = WARM, F1 = "#55A868"),
                        guide = "none") +
      labs(title = "Latest Benchmark Metrics", x = "", y = "Score") +
      theme_clinicalner
  } else {
    p_bm <- ggplot() + theme_void() + labs(title = "Benchmark file unreadable")
  }
} else {
  p_bm <- ggplot() + theme_void() +
    annotate("text", x = 0.5, y = 0.5, label = "No benchmark results.\nRun run_benchmark.py",
             size = 5, colour = "grey50") +
    theme_void() + labs(title = "Benchmark Metrics")
}

## Combine and save
combined_plot <- (p_length | p_scatter) / (p_density | p_bm) +
  plot_annotation(
    title   = "ClinicalNER — Corpus & Model Summary",
    caption = sprintf("Generated: %s", Sys.time()),
    theme   = theme(plot.title = element_text(face = "bold", size = 15))
  )

out_plot_path <- file.path(OUT_DIR, "summary_report.png")
ggsave(out_plot_path, combined_plot, width = 14, height = 10, dpi = 150)
cat(sprintf("Saved plot: %s\n", out_plot_path))

# ── 7. Write R Session Info ───────────────────────────────────────────────────

session_path <- file.path(OUT_DIR, "session_info.txt")
writeLines(capture.output(sessionInfo()), session_path)

cat("\n=== Analysis complete ===\n")
cat(sprintf("Output directory: %s\n", OUT_DIR))
