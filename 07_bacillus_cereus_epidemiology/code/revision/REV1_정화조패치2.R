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

SEPTIC     <- "하수처리구역외_정화조인구"
BREAK_YEAR <- 2017

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
stopifnot(!is.null(PRE_RAW), SEPTIC %in% names(PRE_RAW))


# ── 2. 단절 진단 (지역 분류) ──────────────────────────────────────────────────
cat("\n■ 2. 단절 진단\n")
sp <- data.frame(region = PRE_RAW$region, year = PRE_RAW$year,
                 x = suppressWarnings(as.numeric(PRE_RAW[[SEPTIC]])))
allz <- function(v) any(is.finite(v)) && all(v[is.finite(v)] == 0)
pre_z  <- sp %>% dplyr::filter(year <= BREAK_YEAR - 1) %>% dplyr::group_by(region) %>%
  dplyr::summarise(pre = allz(x), .groups = "drop")
post_z <- sp %>% dplyr::filter(year >= BREAK_YEAR + 1) %>% dplyr::group_by(region) %>%
  dplyr::summarise(post = allz(x), .groups = "drop")
fl <- dplyr::inner_join(pre_z, post_z, by = "region")
SUSPECT <- fl$region[fl$pre & !fl$post]     # 결측-0 의심
TRUEZERO <- fl$region[fl$pre & fl$post]     # 구조적 진짜 0
cat(sprintf("  결측-0 의심 %d개 / 구조적 진짜 0 %d개 / 정상 %d개\n",
            length(SUSPECT), length(TRUEZERO), sum(!fl$pre)))

# ── 3. 패널 D·E ───────────────────────────────────────────────────────────────
cat("\n■ 3. 패널 생성\n")
ZERO_INVALID <- c("상수도보급률","하수도보급률","의사수","관내진료비_전체",
                  "연간인플루엔자예방접종률_표준화율","건강생활실천율_조율",
                  "걷기실천율_표준화율","재정자립도","재정자주도","성비","고령인구비율")
PRE_A <- PRE_RAW
for (v in intersect(ZERO_INVALID, names(PRE_A))) {
  x <- suppressWarnings(as.numeric(PRE_A[[v]])); x[is.finite(x) & x == 0] <- NA_real_; PRE_A[[v]] <- x
}

# D : 결측-0 '의심 95개 지역'의 2011-2017 0만 NA (구조적 0 49개는 보존)
PRE_D <- PRE_A
xd <- suppressWarnings(as.numeric(PRE_D[[SEPTIC]]))
hitD <- is.finite(xd) & xd == 0 & PRE_D$year <= BREAK_YEAR & PRE_D$region %in% SUSPECT
cat(sprintf("  D. 의심지역 한정 교정 : %d 셀 (지난번 B는 1018셀 = 과잉)\n", sum(hitD)))
xd[hitD] <- NA_real_; PRE_D[[SEPTIC]] <- xd

mk <- function(P) { z <- .orig_finalize(P, YS, YE)
  z$area_3 <- sapply(z$region, classify_region)
  z$area_2 <- ifelse(z$area_3 == "농촌", "농촌", "도시"); z }
cm_A <- mk(PRE_A); cm_D <- mk(PRE_D)

# E : 정화조인구를 사전지정 후보에서 제외 (패널은 A 그대로 → N 손실 0)
TV_E <- TV_PRESPEC[TV_PRESPEC$code != SEPTIC, , drop = FALSE]; rownames(TV_E) <- NULL

# ── 4. 적합 + DIC ─────────────────────────────────────────────────────────────
cat("\n■ 4. 적합\n")
fitset <- function(TV, PANEL) list(
  Overall = run_model(TV, PANEL,                                     quiet = TRUE, re_str = NONSPATIAL, fam = PRINCIPAL_FAMILY),
  Urban   = run_model(TV, PANEL %>% dplyr::filter(area_2 == "도시"), quiet = TRUE, re_str = NONSPATIAL, fam = PRINCIPAL_FAMILY),
  Rural   = run_model(TV, PANEL %>% dplyr::filter(area_2 == "농촌"), quiet = TRUE, re_str = NONSPATIAL, fam = PRINCIPAL_FAMILY))
FIT_A <- fitset(TV_PRESPEC, cm_A); cat("  · A 완료\n")
FIT_D <- fitset(TV_PRESPEC, cm_D); cat("  · D 완료\n")
FIT_E <- fitset(TV_E,       cm_A); cat("  · E 완료\n")

# ── 5. 이전 실행(A/B/C) DIC 까지 합쳐서 진단표 ────────────────────────────────
cat("\n■ 5. 적합 진단 (수렴 실패 판별)\n")
diag_row <- function(f, panel, stratum) {
  if (is.null(f$fit)) return(data.frame(panel, stratum, N = NA, n_vars = NA, DIC = NA,
                                        max_abs_logIRR = NA, 판정 = "FIT 실패"))
  fe <- f$fit$summary.fixed; fe <- fe[rownames(fe) != "(Intercept)", , drop = FALSE]
  mal <- max(abs(fe$mean))
  data.frame(panel, stratum, N = nrow(f$ic), n_vars = nrow(f$FMAP), DIC = round(f$dic, 1),
             max_abs_logIRR = round(mal, 3),
             판정 = ifelse(mal < 0.20, "★계수 전면 수축 = 수렴 의심", "정상"))
}
dg <- data.frame()
for (p in c("A","D","E")) for (s in c("Overall","Urban","Rural"))
  dg <- rbind(dg, diag_row(get(paste0("FIT_", p))[[s]], p, s))

old <- list.files(OUTDIR, pattern = "^REV1_정화조교정_objects_.*\\.rds$", full.names = TRUE)
if (length(old)) {
  o <- readRDS(sort(old, decreasing = TRUE)[1])
  for (p in c("FIT_A","FIT_B","FIT_C")) for (s in c("Overall","Urban","Rural"))
    dg <- rbind(dg, diag_row(o[[p]][[s]], paste0("이전_", sub("FIT_", "", p)), s))
}
print(dg, row.names = FALSE)

bind_irr <- function(FIT, tag) dplyr::bind_rows(lapply(names(FIT), function(nm) {
  d <- FIT[[nm]]$mv; if (is.null(d) || !nrow(d)) return(data.frame())
  d$stratum <- nm; d$panel <- tag; d }))
irr_A <- bind_irr(FIT_A, "A_기존교정"); irr_D <- bind_irr(FIT_D, "D_의심지역한정")
irr_E <- bind_irr(FIT_E, "E_정화조제외")

cmp <- data.frame()
for (nm in c("Overall","Urban","Rural")) {
  a <- FIT_A[[nm]]$mv; b <- FIT_D[[nm]]$mv
  if (is.null(a) || is.null(b) || !nrow(a) || !nrow(b)) next
  m <- merge(a[, c("var_kr","형태","IRR","lo","hi","sig")], b[, c("var_kr","형태","IRR","lo","hi","sig")],
             by = "var_kr", all = TRUE, suffixes = c("_A", "_D"))
  m$stratum <- nm
  m$flip <- with(m, ifelse(is.na(sig_A) | is.na(sig_D), "only_one_model",
                    ifelse(sig_A == sig_D, "", "SIGNIFICANCE_CHANGED")))
  cmp <- dplyr::bind_rows(cmp, m)
}
chg <- if (nrow(cmp)) cmp[cmp$flip != "", , drop = FALSE] else cmp
cat(sprintf("\n  ★ A→D 에서 유의성이 바뀐 항목: %d건\n", nrow(chg)))
if (nrow(chg)) print(chg[, c("stratum","var_kr","IRR_A","sig_A","IRR_D","sig_D","flip")], row.names = FALSE)

# ── 6. 저장 ───────────────────────────────────────────────────────────────────
cat("\n■ 6. 저장\n")
xp <- file.path(OUTDIR, sprintf("REV1_정화조교정2_%s.xlsx", TS))
wb <- openxlsx::createWorkbook()
add <- function(nm, df) { if (is.null(df) || !nrow(df)) return(invisible())
  openxlsx::addWorksheet(wb, nm); openxlsx::writeData(wb, nm, df) }
add("21_fit_진단",          dg)
add("22_지역분류",          data.frame(region = fl$region, pre_2011_2016_allzero = fl$pre,
                                    post_2018_2024_allzero = fl$post,
                                    판정 = ifelse(fl$pre & !fl$post, "결측-0 의심",
                                           ifelse(fl$pre & fl$post, "구조적 0", "정상"))))
add("23_IRR_A_기존교정",    irr_A)
add("24_IRR_D_의심지역한정", irr_D)
add("25_IRR_E_정화조제외",  irr_E)
add("26_A_vs_D",            cmp)
openxlsx::saveWorkbook(wb, xp, overwrite = TRUE)
saveRDS(list(FIT_A = FIT_A, FIT_D = FIT_D, FIT_E = FIT_E, dg = dg, cmp = cmp),
        file.path(OUTDIR, sprintf("REV1_정화조교정2_objects_%s.rds", TS)))
cat(sprintf("  ✔ %s\n", xp))
cat(sprintf("\n■ 완료 (%.1f분)\n", as.numeric(difftime(Sys.time(), T_START, units = "mins"))))
cat("\n※ 21_fit_진단 의 '판정' 열을 먼저 보십시오. '수렴 의심'이 찍히면 그 패널 결과는 버립니다.\n")
