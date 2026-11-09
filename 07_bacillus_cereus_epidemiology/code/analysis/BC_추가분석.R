# ==============================================================================
# BC_추가분석.R — IJEHR 투고용 추가 분석 ①~⑤ (데이터 추가 없음, 기존 인프라 재사용)
#   ① 도시 vs 농촌 직접 보정 IRR (pooled, urban 주효과)         → 본문 Table 2 헤드라인
#   ② Table 1 정식 도농 비교 (median[IQR] + Wilcoxon + 조발생률 RR)
#   ③ 모델 보정 점검 (M0 PIT + 영과잉 ZINB 민감도 + PIT 히스토그램)
#   ④ 농촌 parsimonious 모델 (희소표본 안정화) + 상수도 불안정 확인
#   ⑤ 헤드라인 3변수 강건성 표 (Full / COVID제외 / ZINB)
#   실행: source("BC_추가분석.R")   (results/B_cereus/code/ 에서 or 절대경로)
#   ※ 주기간 = 선택된 2011–2024. 다르면 아래 PRINCIPAL_START/END_YEAR 만 바꾸면 됨.
# ==============================================================================
rm(list = ls()); gc()
CODE_DIR <- "/Users/guns/Desktop/도시_농촌_분석/code"
source(file.path(CODE_DIR, "00_config_공통.R"))
source(file.path(CODE_DIR, "01_데이터로드_함수.R"))
source(file.path(CODE_DIR, "02_모델_함수.R"))

DZ <- "바실러스세레우스"; SLUG <- disease_slug(DZ); EN <- DISEASE_EN[DZ]
PRINCIPAL_START <- 2011; END_YEAR <- 2024          # ← 선택된 주기간
CUR_YS <<- PRINCIPAL_START; CUR_YE <<- END_YEAR
TS <- format(Sys.time(), "%y%m%d_%H%M")
OUTDIR <- file.path(DIR_RES, SLUG); SUBMIT <- file.path(OUTDIR, "IJEHR_B_cereus")
dir.create(file.path(SUBMIT, "figures"), recursive = TRUE, showWarnings = FALSE)
NONSPATIAL <- "+ f(idtime,model='rw1',hyper=pc_prec) + f(idarea,model='iid',hyper=pc_prec)"

# ── 공간객체 ──
shp <- st_read(PATH_SHP, quiet = TRUE) %>%
  mutate(region = str_replace_all(as.character(region), "\\s+", ""),
         region = if_else(region == "인천시미추홀구", "인천시남구", region))
shp_main <- shp %>% filter(!region %in% c("인천시옹진군","전라남도완도군","전라남도진도군",
                                          "경상남도거제시","경상남도남해군","경상북도울릉군"))
nb_obj <- poly2nb(shp_main, snap = 0.01); iso <- which(card(nb_obj) == 0)
if (length(iso) > 0) { shp_main <- shp_main[-iso, ]; nb_obj <- poly2nb(shp_main, snap = 0.01) }
nb2INLA(nb_obj, file = "/tmp/bc_add.graph"); g_main <- inla.read.graph("/tmp/bc_add.graph")
nb_w <- nb2listw(nb_obj, style = "W")

pc_bym  <- list(prec.unstruct=list(prior="pc.prec",param=c(0.5,0.01)), prec.spatial=list(prior="pc.prec",param=c(0.5,0.01)))
pc_prec <- list(prec=list(prior="pc.prec",param=c(0.5,0.01)))
# ★ 주모델: ZINB (영과잉 데이터, NB 대비 ΔDIC≈49 우수)
fit_inla <- function(fs, dat, fam="zeroinflatednbinomial1") tryCatch(inla(as.formula(fs), family=fam, data=dat,
  control.compute=list(dic=TRUE, waic=TRUE, cpo=TRUE), control.predictor=list(link=1), verbose=FALSE),
  error=function(e){ cat(sprintf("    ❌ fit(%s): %s\n", fam, e$message)); NULL })
mk_kr <- function(eng_z, FMAP){ rn <- sub("_z$","",eng_z); k <- FMAP$kr[FMAP$eng==rn]; if(length(k)) k[1] else rn }
irr_row <- function(fe, nm) { if (!nm %in% rownames(fe)) return(c(NA,NA,NA,0))
  r <- fe[nm,]; c(exp(r$mean), exp(r$`0.025quant`), exp(r$`0.975quant`), as.integer(r$`0.025quant`>0|r$`0.975quant`<0)) }

# ── 데이터 + M0 준비(ic, FMAP, cov 확보) ──
cm <- build_cor_merged(DZ, PRINCIPAL_START, END_YEAR, include_food = FALSE)
prep <- run_model(TV_PRESPEC, cm, quiet = TRUE, re_str = NONSPATIAL)
ic <- prep$ic; FMAP <- prep$FMAP; cov <- FMAP$safe
ic$urban <- as.integer(ic$area_2 == "도시")        # 도시=1, 농촌=0(기준)
cat(sprintf("\n■ %s | N=%d | 사례=%d | 공변량 %d개\n", EN, nrow(ic), sum(cm$cases), length(cov)))

wb <- createWorkbook()
s_t <- createStyle(fontSize=12, textDecoration="bold"); s_h <- createStyle(textDecoration="bold", halign="center", fgFill="#D9E2F3"); s_s <- createStyle(textDecoration="bold", fontColour="#C00000")

# ══════════════════════════════════════════════════════════════════════════════
# ① 도시 vs 농촌 직접 보정 IRR (pooled, urban 주효과; 도시인구비율은 urban과 중복 → 제외)
# ══════════════════════════════════════════════════════════════════════════════
cat("\n## ① Pooled urban-vs-rural adjusted IRR\n")
cov_pool <- setdiff(cov, grep("urban_pop_rate", cov, value=TRUE))
f_pool <- paste("cases ~", paste(cov_pool, collapse=" + "), "+ urban + offset(log(population+1))", NONSPATIAL)
fit_pool <- fit_inla(f_pool, ic)
pooled_tbl <- data.frame()
if (!is.null(fit_pool)) {
  u <- irr_row(fit_pool$summary.fixed, "urban")
  cat(sprintf("  Urban vs rural (rural=ref) adjusted IRR = %.3f (%.3f–%.3f) %s\n",
      u[1], u[2], u[3], ifelse(u[4]==1,"★","NS")))
  pooled_tbl <- data.frame(Comparison="Urban vs rural (rural=ref)",
    IRR=round(u[1],3), CrI=sprintf("%.3f–%.3f", u[2], u[3]), Sig=ifelse(u[4]==1,"★",""),
    Note="adjusted for all covariates except urban_pop_rate; non-spatial M0 RE")
}
addWorksheet(wb, "A1_Pooled_UrbanRural")
writeData(wb,"A1_Pooled_UrbanRural", sprintf("%s — Adjusted urban-vs-rural IRR (pooled zero-inflated NB model), %d-%d", EN, PRINCIPAL_START, END_YEAR), startRow=1); addStyle(wb,"A1_Pooled_UrbanRural",s_t,rows=1,cols=1)
if (nrow(pooled_tbl)>0) writeData(wb,"A1_Pooled_UrbanRural", pooled_tbl, startRow=3, colNames=TRUE)

# ══════════════════════════════════════════════════════════════════════════════
# ② Table 1 정식 도농 비교 (region-year 단위: median[IQR] + Wilcoxon, + 조발생률 RR)
# ══════════════════════════════════════════════════════════════════════════════
cat("\n## ② Table 1 (descriptive, urban vs rural)\n")
cmg <- cm %>% mutate(grp = ifelse(area_2=="도시","Urban","Rural"))
# (a) 사례/발생률 + 조발생률 RR (NB)
inc <- cmg %>% group_by(grp) %>% summarise(region_years=n(), cases=sum(cases),
        incidence_per100k=round(sum(cases)/sum(population)*1e5,3), .groups="drop")
nb_rr <- tryCatch(MASS::glm.nb(cases ~ urban + offset(log(population+1)),
          data = cm %>% mutate(urban=as.integer(area_2=="도시"))), error=function(e) NULL)
rr_txt <- "NA"
if (!is.null(nb_rr)) { co <- summary(nb_rr)$coefficients
  if ("urban" %in% rownames(co)) { b<-co["urban","Estimate"]; se<-co["urban","Std. Error"]
    rr_txt <- sprintf("%.3f (%.3f–%.3f)", exp(b), exp(b-1.96*se), exp(b+1.96*se)) } }
cat(sprintf("  발생률: Urban %.3f vs Rural %.3f /100k | 조 IRR(urban vs rural) = %s\n",
    inc$incidence_per100k[inc$grp=="Urban"], inc$incidence_per100k[inc$grp=="Rural"], rr_txt))
# (b) 공변량 median[IQR] by group + Wilcoxon (raw 단위)
raw_codes <- intersect(TV_PRESPEC$code, names(cm))
t1 <- data.frame()
for (cc in raw_codes) {
  x <- as.numeric(cmg[[cc]]); g <- cmg$grp
  fmt <- function(v){ v<-v[is.finite(v)]; if(!length(v)) return("—"); sprintf("%.2f [%.2f–%.2f]", median(v), quantile(v,.25), quantile(v,.75)) }
  pw <- tryCatch(wilcox.test(x ~ g)$p.value, error=function(e) NA)
  eng <- TV_PRESPEC$eng[TV_PRESPEC$code==cc][1]
  t1 <- rbind(t1, data.frame(Variable=eng,
      Urban=fmt(x[g=="Urban"]), Rural=fmt(x[g=="Rural"]),
      Wilcoxon_p=ifelse(is.na(pw),"NA",sprintf("%.3f", pw)), stringsAsFactors=FALSE))
}
addWorksheet(wb, "A2_Table1_full")
writeData(wb,"A2_Table1_full", sprintf("%s — Table 1. Descriptive statistics by urban/rural, %d-%d (region-year level; median[IQR])", EN, PRINCIPAL_START, END_YEAR), startRow=1); addStyle(wb,"A2_Table1_full",s_t,rows=1,cols=1)
writeData(wb,"A2_Table1_full", inc, startRow=3, colNames=TRUE); addStyle(wb,"A2_Table1_full",s_h,rows=3,cols=1:ncol(inc),gridExpand=TRUE)
writeData(wb,"A2_Table1_full", sprintf("Crude incidence rate ratio (urban vs rural, NB) = %s", rr_txt), startRow=3+nrow(inc)+1, startCol=1)
writeData(wb,"A2_Table1_full", t1, startRow=3+nrow(inc)+3, colNames=TRUE); addStyle(wb,"A2_Table1_full",s_h,rows=3+nrow(inc)+3,cols=1:ncol(t1),gridExpand=TRUE)

# ══════════════════════════════════════════════════════════════════════════════
# ③ 모델 보정 점검: M0 PIT vs 영과잉 ZINB
# ══════════════════════════════════════════════════════════════════════════════
cat("\n## ③ Model selection: NB vs ZINB (영과잉 검증)\n")
base_f <- paste("cases ~", paste(cov, collapse=" + "), "+ offset(log(population+1))")
fit_nb  <- fit_inla(paste(base_f, NONSPATIAL), ic, fam="nbinomial")
fit_principal <- fit_inla(paste(base_f, NONSPATIAL), ic, fam="zeroinflatednbinomial1")  # ★ 주모델
calib <- data.frame()
get_pit <- function(f){ if(is.null(f)||is.null(f$cpo)) return(c(NA,NA,NA)); p<-f$cpo$pit
  c(mean(p,na.rm=TRUE), sum(f$cpo$failure>0,na.rm=TRUE), ifelse(is.null(f$dic),NA,f$dic$dic)) }
for (nm in c("Negative binomial","Zero-inflated NB")) { f <- if(nm=="Negative binomial") fit_nb else fit_principal; v <- get_pit(f)
  calib <- rbind(calib, data.frame(model=nm, mean_PIT=round(v[1],3), CPO_fail=v[2],
                                   DIC=round(v[3],2), WAIC=ifelse(is.null(f$waic),NA,round(f$waic$waic,2)))) }
cat("  calibration:\n"); print(calib, row.names=FALSE)
addWorksheet(wb, "A3_Calibration"); writeData(wb,"A3_Calibration","Model comparison: NB vs zero-inflated NB (★ ZINB selected: ΔDIC≈49). mean PIT ≈ 0.5 = well-calibrated.", startRow=1); addStyle(wb,"A3_Calibration",s_t,rows=1,cols=1)
writeData(wb,"A3_Calibration", calib, startRow=3, colNames=TRUE); addStyle(wb,"A3_Calibration",s_h,rows=3,cols=1:ncol(calib),gridExpand=TRUE)
# PIT 히스토그램 (보충 그림)
if (!is.null(fit_nb) && !is.null(fit_nb$cpo)) {
  png(file.path(SUBMIT,"figures","FigSx_PIT_histogram.png"), width=1500, height=600, res=180)
  par(mfrow=c(1,2))
  hist(fit_nb$cpo$pit, breaks=20, col="#c0392b", border="white", main="(a) Negative binomial (NB)", xlab="PIT")
  if (!is.null(fit_principal) && !is.null(fit_principal$cpo)) hist(fit_principal$cpo$pit, breaks=20, col="#1f6fb2", border="white", main="(b) Zero-inflated NB (★ selected)", xlab="PIT") else plot.new()
  dev.off(); cat(sprintf("  ★ PIT 히스토그램: %s\n", file.path(SUBMIT,"figures","FigSx_PIT_histogram.png")))
}

# ══════════════════════════════════════════════════════════════════════════════
# ④ 농촌 parsimonious 모델 (희소표본 안정화)
#    full rural = TV_PRESPEC 전체 / parsimonious = forced(성비·고령) + 농촌 유의 이론변수
# ══════════════════════════════════════════════════════════════════════════════
cat("\n## ④ Parsimonious rural model\n")
DAT_rural <- cm %>% filter(area_2=="농촌")
res_rural_full <- run_model(TV_PRESPEC, DAT_rural, quiet=TRUE, re_str=NONSPATIAL)
# 농촌에서 유의했던 이론변수 중심으로 축소 (forced + 지하수·예방접종)
keep_kr <- c("성비","고령인구비율","가정용지하수개소수","인플루엔자예방접종률")
TV_rural_pars <- TV_PRESPEC[TV_PRESPEC$kr %in% keep_kr, ]
res_rural_pars <- run_model(TV_rural_pars, DAT_rural, quiet=TRUE, re_str=NONSPATIAL)
rural_cmp <- function(res, tag){ if(is.null(res$mv)||nrow(res$mv)==0) return(data.frame())
  data.frame(model=tag, Variable=res$mv$var_kr, IRR=res$mv$IRR, lo=res$mv$lo, hi=res$mv$hi, sig=res$mv$sig) }
rural_tbl <- rbind(rural_cmp(res_rural_full,"Rural_full"), rural_cmp(res_rural_pars,"Rural_parsimonious"))
cat("  rural full vs parsimonious (CI 폭 비교):\n"); print(rural_tbl, row.names=FALSE)
addWorksheet(wb, "A4_Rural_parsimonious"); writeData(wb,"A4_Rural_parsimonious", sprintf("%s — Rural model: full vs parsimonious (sparse-sample stabilization). 상수도는 농촌 분산 거의 0 → 제외/각주 권장", EN), startRow=1); addStyle(wb,"A4_Rural_parsimonious",s_t,rows=1,cols=1)
if (nrow(rural_tbl)>0) { writeData(wb,"A4_Rural_parsimonious", rural_tbl, startRow=3, colNames=TRUE); addStyle(wb,"A4_Rural_parsimonious",s_h,rows=3,cols=1:ncol(rural_tbl),gridExpand=TRUE) }

# ══════════════════════════════════════════════════════════════════════════════
# ⑤ 헤드라인 3변수 강건성 표 (Full M0 / COVID제외 / ZINB)
# ══════════════════════════════════════════════════════════════════════════════
cat("\n## ⑤ Robustness of headline findings\n")
head_z <- c("gw_home_count_z","med_total_z","flu_vax_z")   # 가정용지하수·관내진료비·인플루엔자
head_kr <- sapply(head_z, function(z) mk_kr(z, FMAP))
fit_covid <- fit_inla(paste(base_f, NONSPATIAL), ic %>% filter(!year %in% c(2020,2021)), fam="zeroinflatednbinomial1")  # ★ ZINB
spec_irr <- function(fit, z){ if(is.null(fit)) return("—"); v<-irr_row(fit$summary.fixed, z)
  if(is.na(v[1])) return("—"); sprintf("%.2f (%.2f–%.2f)%s", v[1],v[2],v[3], ifelse(v[4]==1,"*","")) }
rob <- data.frame(Variable=head_kr,
  Full_ZINB   = sapply(head_z, function(z) spec_irr(fit_principal, z)),
  COVID_excl_ZINB = sapply(head_z, function(z) spec_irr(fit_covid, z)),
  NB_for_comparison = sapply(head_z, function(z) spec_irr(fit_nb, z)), row.names=NULL)
cat("  headline robustness (* = 95% CrI excludes 1):\n"); print(rob, row.names=FALSE)
addWorksheet(wb, "A5_Robustness_headline"); writeData(wb,"A5_Robustness_headline", sprintf("%s — Robustness of headline covariates across specifications (* = 95%% CrI excludes 1; all ZINB model)", EN), startRow=1); addStyle(wb,"A5_Robustness_headline",s_t,rows=1,cols=1)
writeData(wb,"A5_Robustness_headline", "Note: Full_ZINB = main model; COVID_excl_ZINB = sensitivity (years 2020-21 removed); NB_for_comparison = shows robustness vs model choice. No multiplicity adjustment → interpret robust effects as supporting evidence.", startRow=2)
writeData(wb,"A5_Robustness_headline", rob, startRow=4, colNames=TRUE); addStyle(wb,"A5_Robustness_headline",s_h,rows=4,cols=1:ncol(rob),gridExpand=TRUE)

# ── 저장 ──
fn <- file.path(SUBMIT, sprintf("%s_ADDON_%dto%d_%s.xlsx", SLUG, PRINCIPAL_START, END_YEAR, TS))
saveWorkbook(wb, fn, overwrite=TRUE)
cat(sprintf("\n✅ 추가분석 완료\n   ★ %s\n   ★ figures/FigSx_PIT_histogram.png\n", fn))
