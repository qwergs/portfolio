# ==============================================================================
# 00_export_figure_data.R — figure inputs that only R can build
#
#   Run once:  source("figures/00_export_figure_data.R", encoding = "UTF-8")
#   Writes  :  <B_cereus>/figures/_data/
#                panel_district_year.csv   (Figure 1, S2, S3b, S3c)
#                moran.csv                 (Figure S2g)
#                monthly_cases.csv         (Figure S2a, S2b)
#
#   Then     :  python3 figures/make_all_figures.py
#
#   It rebuilds exactly the panel the manuscript uses (M2 = cm_P1: the
#   urban-area population rate above 100% in 2020 set to missing), using the
#   ZENODO deposit code, the same as REV1_M2_적합도.R. Nothing is refitted,
#   so it takes seconds, not minutes.
# ==============================================================================

suppressPackageStartupMessages({
  library(dplyr); library(sf); library(spdep); library(stringr)
})

# ── 0. paths (identical to the REV1 scripts) ──────────────────────────────────
ROOT     <- "/Users/guns/Desktop/과거자료/학부생_전처리_논문/도시_농촌_분석"
OLD_ROOT <- "/Users/guns/Desktop/도시_농촌_분석"
BASE     <- file.path(ROOT, "B_cereus")
OUTDIR   <- file.path(BASE, "figures", "_data")
dir.create(OUTDIR, recursive = TRUE, showWarnings = FALSE)

USE_SUBMITTED_CODE <- TRUE          # TRUE = ZENODO deposit (as submitted)
CODEDIR  <- if (USE_SUBMITTED_CODE)
  file.path(BASE, "기타", "ZENODO_DEPOSIT", "code") else file.path(ROOT, "code")
OLD_DATA <- "/Users/guns/Desktop/건상_시각화/FBD_DATA_ZIP"
NEW_DATA <- file.path(ROOT, "FBD_DATA_ZIP")
stopifnot(dir.exists(CODEDIR), dir.exists(NEW_DATA))

src_patched <- function(fname) {
  fp  <- file.path(CODEDIR, fname); stopifnot(file.exists(fp))
  txt <- readLines(fp, encoding = "UTF-8", warn = FALSE)
  txt <- gsub(OLD_DATA, NEW_DATA, txt, fixed = TRUE)
  txt <- gsub(OLD_ROOT, ROOT,     txt, fixed = TRUE)
  eval(parse(text = paste(txt, collapse = "\n")), envir = globalenv())
  cat(sprintf("  · source: %s\n", fname))
}
cat("■ 0. source\n")
src_patched("00_config_공통.R")
src_patched("01_데이터로드_함수.R")
src_patched("02_모델_함수.R")

DZ <- "바실러스세레우스"; YS <- 2011; YE <- 2024

# ── 1. spatial object and the 223 modelled districts ──────────────────────────
cat("\n■ 1. spatial\n")
shp <- sf::st_read(PATH_SHP, quiet = TRUE) %>%
  mutate(region = stringr::str_replace_all(as.character(region), "\\s+", ""),
         region = dplyr::if_else(region == "인천시미추홀구", "인천시남구", region))
EXCLUDED_ISLANDS <- c("인천시옹진군","전라남도완도군","전라남도진도군",
                      "경상남도거제시","경상남도남해군","경상북도울릉군")
shp_main <- shp %>% filter(!region %in% EXCLUDED_ISLANDS)
nb_obj <- spdep::poly2nb(shp_main, snap = 0.01)
iso <- which(spdep::card(nb_obj) == 0)
if (length(iso) > 0) { shp_main <- shp_main[-iso, ]; nb_obj <- spdep::poly2nb(shp_main, snap = 0.01) }
nb_w <- spdep::nb2listw(nb_obj, style = "W")
cat(sprintf("  · modelled districts: %d of %d\n", nrow(shp_main), nrow(shp)))

# ── 2. panel (M2 = cm_P1) ─────────────────────────────────────────────────────
cat("\n■ 2. panel\n")
CUR_YS <<- YS; CUR_YE <<- YE
PRE_RAW <- NULL
.orig_finalize <- finalize_covariates
finalize_covariates <- function(cm, ys, ye) { PRE_RAW <<- cm; .orig_finalize(cm, ys, ye) }
invisible(build_cor_merged(DZ, YS, YE, include_food = FALSE))
finalize_covariates <- .orig_finalize
stopifnot(!is.null(PRE_RAW))

URBAN <- "도시지역인구비율"
PRE_P1 <- PRE_RAW
xu  <- suppressWarnings(as.numeric(PRE_P1[[URBAN]]))
bad <- is.finite(xu) & xu > 100
cat(sprintf("  · %s > 100 set to NA: %d cells\n", URBAN, sum(bad)))
xu[bad] <- NA_real_; PRE_P1[[URBAN]] <- xu

cm_P1 <- .orig_finalize(PRE_P1, YS, YE)
cm_P1$area_3 <- sapply(cm_P1$region, classify_region)
cm_P1$area_2 <- ifelse(cm_P1$area_3 == "농촌", "농촌", "도시")

f <- file.path(OUTDIR, "panel_district_year.csv")
readr_ok <- requireNamespace("readr", quietly = TRUE)
if (readr_ok) readr::write_csv(cm_P1, f) else
  write.csv(cm_P1, f, row.names = FALSE, fileEncoding = "UTF-8")
cat(sprintf("  · %s  (%d rows, %d cols)\n", basename(f), nrow(cm_P1), ncol(cm_P1)))

# ── 3. Moran's I on the modelled districts ────────────────────────────────────
cat("\n■ 3. Moran\n")
rate <- cm_P1 %>% filter(region %in% shp_main$region) %>%
  group_by(region) %>%
  summarise(rate = sum(cases) / sum(population) * 1e5, .groups = "drop")
rate <- rate[match(shp_main$region, rate$region), ]
#   p_analytic is the value reported in the manuscript and in Supplementary
#   Table S9: spdep::moran.test, randomisation assumption, alternative =
#   "greater". p_mc is the 999-permutation p and is kept for reference only;
#   do not report it, the two are not interchangeable.
mt   <- spdep::moran.test(rate$rate, nb_w, randomisation = TRUE)
mc   <- spdep::moran.mc(rate$rate, nb_w, nsim = 999)
mor  <- data.frame(region = rate$region, rate = rate$rate,
                   lag = spdep::lag.listw(nb_w, rate$rate),
                   I = as.numeric(mt$estimate[1]),
                   p_analytic = mt$p.value, p_mc = mc$p.value)
write.csv(mor, file.path(OUTDIR, "moran.csv"), row.names = FALSE, fileEncoding = "UTF-8")
cat(sprintf("  · Moran's I = %.4f, p = %.3f (moran.test, randomisation, greater)\n",
            mor$I[1], mor$p_analytic[1]))
cat(sprintf("    999-permutation p = %.3f (reference only)\n", mor$p_mc[1]))

# ── 4. monthly cases (Figure S2a, S2b) ────────────────────────────────────────
#   식중독최종.csv column names are probed rather than assumed; if no month
#   information is found the file is skipped and Figure S2a/b are left blank.
cat("\n■ 4. monthly\n")
ok <- FALSE
try({
  raw <- DF_RAW_CACHE
  nm  <- names(raw)
  dz_col <- nm[grepl("병원체|질병|disease|균", nm)][1]
  yr_col <- nm[grepl("^연도$|^year$|발생연도", nm, ignore.case = TRUE)][1]
  mo_col <- nm[grepl("^월$|month|발생월", nm, ignore.case = TRUE)][1]
  dt_col <- nm[grepl("발생일|일자|날짜|date", nm, ignore.case = TRUE)][1]
  n_col  <- nm[grepl("환자수|건수|cases|count", nm, ignore.case = TRUE)][1]

  d <- raw
  if (!is.na(dz_col)) d <- d[grepl(DZ, d[[dz_col]]), , drop = FALSE]
  if (is.na(mo_col) && !is.na(dt_col)) {
    dd <- as.Date(d[[dt_col]]); d$..mo <- as.integer(format(dd, "%m"))
    if (is.na(yr_col)) { d$..yr <- as.integer(format(dd, "%Y")); yr_col <- "..yr" }
    mo_col <- "..mo"
  }
  stopifnot(!is.na(mo_col), !is.na(yr_col))
  d$..n <- if (!is.na(n_col)) suppressWarnings(as.numeric(d[[n_col]])) else 1
  out <- d %>%
    transmute(year = as.integer(.data[[yr_col]]),
              month = as.integer(.data[[mo_col]]), cases = ..n) %>%
    filter(year >= YS, year <= YE, month >= 1, month <= 12) %>%
    group_by(year, month) %>% summarise(cases = sum(cases, na.rm = TRUE),
                                        .groups = "drop")
  write.csv(out, file.path(OUTDIR, "monthly_cases.csv"),
            row.names = FALSE, fileEncoding = "UTF-8")
  cat(sprintf("  · monthly_cases.csv  (%d rows, %d cases)\n",
              nrow(out), sum(out$cases)))
  ok <- TRUE
}, silent = TRUE)
if (!ok) {
  cat("  ! no month column found in 식중독최종.csv; Figure S2a/S2b will be blank.\n")
  cat("    columns were:\n      ", paste(names(DF_RAW_CACHE), collapse = ", "), "\n")
}

cat("\n완료. 이제 실행:  python3 ", file.path(BASE, "figures", "make_all_figures.py"), "\n")
