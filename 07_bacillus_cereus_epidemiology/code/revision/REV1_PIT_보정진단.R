# ==============================================================================
#  REV1 §4-G : 보정(calibration) 진단 - 비무작위 PIT vs mid-PIT
#  ------------------------------------------------------------------------
#  원고는 mean PIT = 0.97 을 "well calibrated" 의 근거로 제시하지만,
#  PIT 평균은 잘 보정된 모형에서 0.5 에 가까워야 합니다.
#  INLA 가 반환하는 cpo$pit 은 이산자료에서 비무작위 PIT = P(Y <= y) 이고,
#  관측의 97%가 0 이므로 y=0 인 칸의 PIT 는 곧 P(Y=0) ~ 0.96 이 됩니다.
#  즉 0.97 은 보정 불량이 아니라 이산·영과잉 자료의 필연적 산물입니다.
#  표준 해법은 mid-PIT = PIT - 0.5*CPO 이며, 잘 보정된 모형에서 평균 ~0.5.
#
#  실행:  Rscript -e 'source("REV1_PIT_보정진단.R", encoding="UTF-8")'
# ==============================================================================

rm(list = ls()); gc()

# ── 0. 경로 + 소스 로드 (원본 스크립트와 동일) ────────────────────────────────
ROOT     <- "/Users/guns/Desktop/과거자료/학부생_전처리_논문/도시_농촌_분석"
OLD_ROOT <- "/Users/guns/Desktop/도시_농촌_분석"
OUTDIR   <- file.path(ROOT, "B_cereus", "IJEHR_REV1")
if (!dir.exists(ROOT)) stop("ROOT 경로가 없습니다: ", ROOT)
if (!dir.exists(OUTDIR)) dir.create(OUTDIR, recursive = TRUE)

# ★ 2026-09-15: 투고본(5변환) 코드로 고정.
#   ROOT/code/02_모델_함수.R 은 2026-07-09 에 5변환 → 8변환으로 개정되었고
#   성비 raw 특례·사육두수 per10k 특례가 제거되었습니다(파일 mtime·헤더 주석으로 확인).
#   투고 원고(MAIN 260620_2016.xlsx, 2026-06-23)는 개정 이전 코드로 산출되었으므로,
#   리비전 민감도 분석은 반드시 투고 당시 코드 = ZENODO 기탁본으로 적합해야 합니다.
#   FALSE 로 두면 현행(8변환) 코드를 씁니다. 원고 재현에는 TRUE 가 맞습니다.
USE_SUBMITTED_CODE <- TRUE
CODEDIR  <- if (USE_SUBMITTED_CODE) {
  file.path(ROOT, "B_cereus", "기타", "ZENODO_DEPOSIT", "code")
} else {
  file.path(ROOT, "code")
}
OLD_DATA <- "/Users/guns/Desktop/건상_시각화/FBD_DATA_ZIP"   # 기탁본에 박힌 옛 데이터 경로
NEW_DATA <- file.path(ROOT, "FBD_DATA_ZIP")
if (!dir.exists(CODEDIR)) stop("CODEDIR 없음: ", CODEDIR)
if (!dir.exists(NEW_DATA)) stop("데이터 폴더 없음: ", NEW_DATA)
cat(sprintf("  · 코드 출처: %s\n", ifelse(USE_SUBMITTED_CODE, "ZENODO 기탁본 (투고본 5변환)", "현행 작업본 (8변환)")))

src_patched <- function(fname) {
  fp <- file.path(CODEDIR, fname)
  if (!file.exists(fp)) stop("소스 없음: ", fp)
  txt <- readLines(fp, encoding = "UTF-8", warn = FALSE)
  txt <- gsub(OLD_DATA, NEW_DATA, txt, fixed = TRUE)
  txt <- gsub(OLD_ROOT, ROOT,     txt, fixed = TRUE)
  eval(parse(text = paste(txt, collapse = "\n")), envir = globalenv())
  cat(sprintf("  · source: %s\n", fname))
}
cat("■ 0. 소스 로드\n")
src_patched("00_config_공통.R")
src_patched("01_데이터로드_함수.R")
src_patched("02_모델_함수.R")

DZ <- "바실러스세레우스"
YS <- 2011; YE <- 2024
TS <- format(Sys.time(), "%y%m%d_%H%M")
NONSPATIAL       <- "+ f(idtime,model='rw1',hyper=pc_prec) + f(idarea,model='iid',hyper=pc_prec)"
PRINCIPAL_FAMILY <- "zeroinflatednbinomial1"
pc_prec <- list(prec = list(prior = "pc.prec", param = c(0.5, 0.01)))


# ── 1. 공간객체 (원본 스크립트 §1 그대로) (주분석과 동일) ───────────────────────────────────────────────
cat("\n■ 1. 공간객체\n")
shp <- sf::st_read(PATH_SHP, quiet = TRUE) %>%
  mutate(region = stringr::str_replace_all(as.character(region), "\\s+", ""),
         region = dplyr::if_else(region == "인천시미추홀구", "인천시남구", region))
EXCLUDED_ISLANDS <- c("인천시옹진군","전라남도완도군","전라남도진도군",
                      "경상남도거제시","경상남도남해군","경상북도울릉군")
shp_main <- shp %>% filter(!region %in% EXCLUDED_ISLANDS)
nb_obj <- spdep::poly2nb(shp_main, snap = 0.01)
iso <- which(spdep::card(nb_obj) == 0)
ISO_REGIONS <- if (length(iso) > 0) shp_main$region[iso] else character(0)
if (length(iso) > 0) { shp_main <- shp_main[-iso, ]; nb_obj <- spdep::poly2nb(shp_main, snap = 0.01) }
spdep::nb2INLA(nb_obj, file = file.path(tempdir(), "bc_all.graph"))
g_main <- INLA::inla.read.graph(file.path(tempdir(), "bc_all.graph"))
nb_w <- spdep::nb2listw(nb_obj, style = "W")
cat(sprintf("  모형 대상 시군구: %d개 (전체 %d개 중 도서·비인접 %d개 제외)\n",
            nrow(shp_main), nrow(shp), nrow(shp) - nrow(shp_main)))
cat(sprintf("  제외: %s\n", paste(c(EXCLUDED_ISLANDS, ISO_REGIONS), collapse = ", ")))

# ── 1b. 패널 생성 (PRE_RAW 후킹) ───────────────────────────────────────────────
cat("\n■ 1b. 패널 생성\n")
CUR_YS <<- YS; CUR_YE <<- YE
PRE_RAW <- NULL
.orig_finalize <- finalize_covariates
finalize_covariates <- function(cm, ys, ye) { PRE_RAW <<- cm; .orig_finalize(cm, ys, ye) }
cm <- build_cor_merged(DZ, YS, YE, include_food = FALSE)
finalize_covariates <- .orig_finalize
stopifnot(!is.null(PRE_RAW))



# ── 2. NB vs ZINB 적합 + PIT 진단 ─────────────────────────────────────────────
cat("\n■ 2. 보정 진단 (비무작위 PIT vs mid-PIT)\n")
cm$area_3 <- sapply(cm$region, classify_region)
cm$area_2 <- ifelse(cm$area_3 == "농촌", "농촌", "도시")

# ── 도우미 ───────────────────────────────────────────────────────────────────
#  INLA cpo$pit = F(y_i) = P(Y <= y_i),  cpo$cpo = P(Y = y_i)
#  따라서 F(y_i - 1) = pit - cpo.
#  · mid-PIT        = pit - 0.5*cpo                (스칼라 요약. 보정되면 평균 ~0.5)
#  · 무작위 PIT     = F(y-1) + v*(F(y)-F(y-1)), v~U(0,1)   (KS 검정용)
#  · 비무작위 PIT 히스토그램 (Czado/Gneiting/Held 2009):
#      F_bar(u) = mean_i clamp((u - F(y_i-1)) / (F(y_i) - F(y_i-1)))
#      j번째 막대 높이 = J * [F_bar(j/J) - F_bar((j-1)/J)]  → 보정되면 전부 ~1
.pit_parts <- function(fit) {
  ok  <- which(is.finite(fit$cpo$pit) & is.finite(fit$cpo$cpo) &
               (is.null(fit$cpo$failure) | fit$cpo$failure == 0))
  pit <- pmin(1, pmax(0, fit$cpo$pit[ok])); cpo <- pmin(1, pmax(0, fit$cpo$cpo[ok]))
  lo  <- pmax(0, pit - cpo); hi <- pmax(lo, pit)
  list(pit = pit, cpo = cpo, lo = lo, hi = hi, n = length(ok))
}
pit_hist <- function(P, J = 20) {
  u <- seq(0, 1, length.out = J + 1)
  Fb <- vapply(u, function(uu)
    mean(pmin(1, pmax(0, (uu - P$lo) / pmax(P$hi - P$lo, 1e-12)))), numeric(1))
  J * diff(Fb)
}
rand_pit <- function(P, seed = 20260915) {
  set.seed(seed); P$lo + runif(P$n) * (P$hi - P$lo)
}

pit_row <- function(fit, stratum, fam) {
  if (is.null(fit) || is.null(fit$cpo)) return(data.frame())
  P <- .pit_parts(fit); mid <- P$pit - 0.5 * P$cpo; rp <- rand_pit(P); h <- pit_hist(P, 20)
  data.frame(stratum = stratum, likelihood = fam, n = P$n,
             DIC = round(fit$dic$dic, 2), WAIC = round(fit$waic$waic, 2),
             mean_PIT_nonrandomised = round(mean(P$pit), 4),
             mean_CPO               = round(mean(P$cpo), 4),
             mean_PIT_mid           = round(mean(mid), 4),
             mean_PIT_randomised    = round(mean(rp), 4),
             hist_max_dev_from_1    = round(max(abs(h - 1)), 3),
             hist_rmse_from_1       = round(sqrt(mean((h - 1)^2)), 3),
             ks_p_randomised = tryCatch(round(suppressWarnings(
                 ks.test(rp, "punif")$p.value), 4), error = function(e) NA))
}

FAMS <- c(ZINB = "zeroinflatednbinomial1", NB = "nbinomial")
PANELS <- list(Overall = cm,
               Urban   = cm %>% dplyr::filter(area_2 == "도시"),
               Rural   = cm %>% dplyr::filter(area_2 == "농촌"))
out <- data.frame(); MIDS <- list()
for (nm in names(PANELS)) for (fk in names(FAMS)) {
  r <- run_model(TV_PRESPEC, PANELS[[nm]], quiet = TRUE,
                 re_str = NONSPATIAL, fam = FAMS[[fk]])
  out <- rbind(out, pit_row(r$fit, nm, fk))
  if (!is.null(r$fit) && !is.null(r$fit$cpo)) {
    P <- .pit_parts(r$fit)
    MIDS[[paste(nm, fk)]] <- list(raw = P$pit, mid = P$pit - 0.5 * P$cpo,
                                  rnd = rand_pit(P), h = pit_hist(P, 20))
  }
  cat(sprintf("  · %s / %s 완료\n", nm, fk))
}
print(out, row.names = FALSE)

cat("\n  [해석]\n")
cat("   · 비무작위 PIT 평균이 1 근처인 것은 이산·영과잉 자료에서 정상입니다.\n")
cat("   · mid-PIT 평균 ~0.5 = 보정 적절. 단 mid-PIT 자체는 한 점에 몰리므로 그림에 쓰지 마십시오.\n")
cat("   · 그림·검정용은 비무작위 PIT 히스토그램(막대 높이 전부 ~1) 과 무작위 PIT(KS 검정) 입니다.\n")
cat("   · hist_max_dev_from_1 이 작을수록 균일에 가깝습니다.\n")

# ── 3. 보충 그림 (Overall, ZINB vs NB) ────────────────────────────────────────
png(file.path(OUTDIR, sprintf("FigS2_PIT_%s.png", TS)),
    width = 2700, height = 900, res = 200)
par(mfrow = c(1, 3), mar = c(4.4, 4.4, 3.0, 1))
Z <- MIDS[["Overall ZINB"]]; N <- MIDS[["Overall NB"]]
hist(Z$raw, breaks = seq(0, 1, 0.05), freq = FALSE, col = "grey80", border = "white",
     main = "(a) Non-randomised PIT values\n(concentrated near 1 by construction)",
     xlab = "P(Y <= y)", ylab = "Density", xlim = c(0, 1))
abline(h = 1, lty = 2)
mids <- seq(0.025, 0.975, by = 0.05)
barplot(Z$h, names.arg = sprintf("%.2f", mids), space = 0, col = "grey55", border = "white",
        ylim = c(0, max(1.6, max(Z$h) * 1.15)), las = 2, cex.names = 0.6,
        main = "(b) PIT histogram, zero-inflated NB", xlab = "u", ylab = "Relative frequency")
abline(h = 1, lty = 2, lwd = 2); box()
barplot(N$h, names.arg = sprintf("%.2f", mids), space = 0, col = "grey75", border = "white",
        ylim = c(0, max(1.6, max(N$h) * 1.15)), las = 2, cex.names = 0.6,
        main = "(c) PIT histogram, negative binomial", xlab = "u", ylab = "Relative frequency")
abline(h = 1, lty = 2, lwd = 2); box()
dev.off()

hist_tab <- data.frame(bin_mid = mids)
for (k in names(MIDS)) hist_tab[[gsub(" ", "_", k)]] <- round(MIDS[[k]]$h, 4)

xp <- file.path(OUTDIR, sprintf("REV1_PIT진단_%s.xlsx", TS))
wb <- openxlsx::createWorkbook()
openxlsx::addWorksheet(wb, "27_PIT_calibration"); openxlsx::writeData(wb, "27_PIT_calibration", out)
openxlsx::addWorksheet(wb, "28_PIT_histogram"); openxlsx::writeData(wb, "28_PIT_histogram", hist_tab)
openxlsx::saveWorkbook(wb, xp, overwrite = TRUE)
cat(sprintf("\n  ✔ %s\n", xp))
cat(sprintf("  ✔ %s\n", file.path(OUTDIR, sprintf("FigS2_PIT_%s.png", TS))))
cat("\n※ Overall/ZINB 행의 mean_PIT_mid 값을 대응문 §4-G 의 0.4X 자리에,\n")
cat("   Overall/NB 행의 값을 0.4Y 자리에 넣으십시오.\n")
