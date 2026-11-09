# ==============================================================================
# BC_도농_통합분석.R — Bacillus cereus 도시/농촌 통합 분석 (내 코드 + 교수님 코드 혼합)
#
#   설계 결정(사용자 확정):
#     · 기간 스윕: 시작연도 2011→2015 (끝 2024 고정) 5개 기간 모두 적합 → 비교표 →
#       '최장기간 우선 + 결과 안정성 + 모델적합(DIC/WAIC)'으로 주기간(principal) 선택.
#       (2015–2024 = 10년이 최소/마지노선, 2011–2024 = 14년이 최장)
#     · 변수방식 = NORO 정직판(PRE-SPECIFIED): 형태탐색 + p<0.20 + VIF<10,
#       강제보정 = 성비·고령(2개). 5개 기간 전부 동일 방식 (TV_PRESPEC 고정 21개).
#     · 비공간 질병 → M0(비공간) + M1–M6(공간) 사다리 비교 후 비공간 채택 규칙.
#     · 본문 디스플레이 = 4개(표+그림 합산). 그 외는 전부 보충자료.
#
#   교수님 코드에서 가져온 것(혼합 지점):
#     · classify_region / run_model / build_cor_merged / moran_prescreen (NORO 통합full 일반화)
#     · 모델비교 M-사다리, 사전/사후 Moran (NORO·HAV)
#     · [보충] CPO/PIT 예측진단, 사례 vs 건수(outbreak) 단위 민감도, 8그래프는 비공간이라 생략
#     · [내 코드] 2011–2024 장기 + 다년보간/스팬게이트, 도농 상호작용 주분석화
#
#   실행: source("BC_도농_통합분석.R")
#   ※ INLA 단일스레드(00_config에서 num.threads='1:1') → 재실행 시 동일 결과(일관성).
# ==============================================================================
rm(list = ls()); gc()
CODE_DIR <- "/Users/guns/Desktop/도시_농촌_분석/code"
source(file.path(CODE_DIR, "00_config_공통.R"))
source(file.path(CODE_DIR, "01_데이터로드_함수.R"))
source(file.path(CODE_DIR, "02_모델_함수.R"))

DZ <- "바실러스세레우스"; SLUG <- disease_slug(DZ); EN <- DISEASE_EN[DZ]
TS <- format(Sys.time(), "%y%m%d_%H%M")
OUTDIR <- file.path(DIR_RES, SLUG); if (!dir.exists(OUTDIR)) dir.create(OUTDIR, recursive = TRUE)
SUBMIT <- file.path(OUTDIR, "IJEHR_B_cereus")            # 최종 제출 폴더(저널명_질병명)
if (!dir.exists(SUBMIT)) dir.create(file.path(SUBMIT, "figures", "tiff"), recursive = TRUE)

# ── 기간 스윕 정의 (끝 2024 고정) ──
SWEEP_STARTS <- 2011:2015                 # 최장 2011 … 마지노선 2015
END_YEAR     <- 2024
NONSPATIAL   <- "+ f(idtime,model='rw1',hyper=pc_prec) + f(idarea,model='iid',hyper=pc_prec)"
# 수동 지정 시 여기에 시작연도를 넣으면 자동선택을 덮어씀 (예: PRINCIPAL_START_OVERRIDE <- 2013)
PRINCIPAL_START_OVERRIDE <- NA
# ── 주모델 우도(likelihood) ──
#   기간 스윕·M0~M6 공간사다리 = NB(빠르고 공간성 판정 일관) / 주기간 본분석 = ZINB(영과잉, 적합 우수)
#   ※ ADDON 분석에서 ZINB가 NB 대비 ΔDIC≈49 우수 → 주모델 ZINB 채택, NB는 비교로 병기.
PRINCIPAL_FAMILY <- "zeroinflatednbinomial1"

# ── 공간객체 (질병/기간 무관, 1회 생성) ──
shp <- st_read(PATH_SHP, quiet = TRUE) %>%
  mutate(region = str_replace_all(as.character(region), "\\s+", ""),
         region = if_else(region == "인천시미추홀구", "인천시남구", region))
shp_main <- shp %>% filter(!region %in% c("인천시옹진군","전라남도완도군","전라남도진도군",
                                          "경상남도거제시","경상남도남해군","경상북도울릉군"))
nb_obj <- poly2nb(shp_main, snap = 0.01); iso <- which(card(nb_obj) == 0)
if (length(iso) > 0) { shp_main <- shp_main[-iso, ]; nb_obj <- poly2nb(shp_main, snap = 0.01) }
nb2INLA(nb_obj, file = "/tmp/bc_all.graph"); g_main <- inla.read.graph("/tmp/bc_all.graph")
nb_w <- nb2listw(nb_obj, style = "W")
cat(sprintf("\n■ %s (%s) 도농 통합분석 | 분석 시군구 %d개\n", DZ, EN, nrow(shp_main)))

pc_bym  <- list(prec.unstruct=list(prior="pc.prec",param=c(0.5,0.01)), prec.spatial=list(prior="pc.prec",param=c(0.5,0.01)))
pc_prec <- list(prec=list(prior="pc.prec",param=c(0.5,0.01)))

# ── 공통 INLA 적합기 / 잔차 Moran ──
fit_inla <- function(fs, dat, fam=PRINCIPAL_FAMILY) tryCatch(inla(as.formula(fs), family=fam, data=dat,
  control.compute=list(dic=TRUE, waic=TRUE, cpo=TRUE), control.predictor=list(link=1), verbose=FALSE),
  error=function(e){ cat(sprintf("    ❌ fit: %s\n", e$message)); NULL })

resid_moran <- function(fit, ic) {
  if (is.null(fit) || is.null(fit$summary.fitted.values)) return(c(NA, NA))
  r <- ic$cases - fit$summary.fitted.values$mean[1:nrow(ic)]
  rdf <- data.frame(region=ic$region, r=r) %>% group_by(region) %>% summarise(r=mean(r), .groups="drop")
  rv <- rdf$r[match(shp_main$region, rdf$region)]; rv[is.na(rv)] <- 0
  m <- tryCatch(moran.test(rv, nb_w), error=function(e) NULL)
  if (is.null(m)) c(NA, NA) else c(unname(m$estimate[1]), m$p.value)
}

# ── M0(비공간) + M1–M6(공간) 사다리 정의 ──
RE_LADDER <- list(
  "M0_nonspatial" = NONSPATIAL,
  "M1_NBonly"     = "",
  "M2_ICAR"       = "+ f(idarea,model='besag',graph=g_main,scale.model=TRUE,hyper=pc_prec)",
  "M3_BYM"        = "+ f(idarea,model='bym',graph=g_main,scale.model=TRUE,hyper=pc_bym)",
  "M4_BYM_IID"    = "+ f(idarea,model='bym',graph=g_main,scale.model=TRUE,hyper=pc_bym) + f(idarea_time,model='iid',hyper=pc_prec)",
  "M5_BYM_RW1"    = "+ f(idarea,model='bym',graph=g_main,scale.model=TRUE,hyper=pc_bym) + f(idtime,model='rw1',hyper=pc_prec)",
  "M6_BYM_RW1_IID"= "+ f(idarea,model='bym',graph=g_main,scale.model=TRUE,hyper=pc_bym) + f(idtime,model='rw1',hyper=pc_prec) + f(idarea_time,model='iid',hyper=pc_prec)")

mk_kr <- function(eng_z, FMAP){ rn <- sub("_z$","",eng_z); k <- FMAP$kr[FMAP$eng==rn]; if(length(k)) k[1] else rn }

# ══════════════════════════════════════════════════════════════════════════════
# 한 기간 분석
#   full=FALSE : 스윕용(경량) — M0·M6만 적합 + 3그룹 유의수 + 사전/사후 Moran
#   full=TRUE  : 주기간용(전체) — M0~M6 전체 + 3그룹 IRR + 변수별 상호작용 + CPO
# ══════════════════════════════════════════════════════════════════════════════
analyze_period <- function(ys, ye, full = FALSE, fam = FAMILY) {
  CUR_YS <<- ys; CUR_YE <<- ye
  cm <- build_cor_merged(DZ, ys, ye, include_food = FALSE)
  mor <- moran_prescreen(cm)
  prep <- run_model(TV_PRESPEC, cm, quiet = TRUE, re_str = NONSPATIAL)
  if (is.null(prep$ic) || is.null(prep$FMAP)) return(list(ok=FALSE, ys=ys, ye=ye, cm=cm, mor=mor))
  ic <- prep$ic; FMAP <- prep$FMAP; cov <- FMAP$safe
  ic$urban <- as.integer(ic$area_2 == "도시")
  base_f <- paste("cases ~", paste(cov, collapse=" + "), "+ offset(log(population+1))")

  # ── 모델비교 (스윕: M0·M6 / 전체: M0~M6) — NB로 빠르게 적합·공간성 판정 ──
  use_models <- if (full) names(RE_LADDER) else c("M0_nonspatial","M6_BYM_RW1_IID")
  fits <- list(); cmp <- data.frame()
  for (nm in use_models) {
    f <- fit_inla(paste(base_f, RE_LADDER[[nm]]), ic, fam="nbinomial"); fits[[nm]] <- f
    if (is.null(f)) { cmp <- rbind(cmp, data.frame(model=nm, DIC=NA, WAIC=NA, postMoranI=NA, postMoranP=NA)); next }
    pm <- resid_moran(f, ic)
    cmp <- rbind(cmp, data.frame(model=nm, DIC=round(f$dic$dic,2), WAIC=round(f$waic$waic,2),
                                 postMoranI=round(pm[1],4), postMoranP=round(pm[2],4)))
  }
  cmp$dDIC <- round(cmp$DIC - min(cmp$DIC, na.rm=TRUE), 2)
  dic_m0 <- cmp$DIC[cmp$model=="M0_nonspatial"]; dic_m6 <- cmp$DIC[cmp$model=="M6_BYM_RW1_IID"]
  best_nm <- if (!is.na(dic_m0) && !is.na(dic_m6) && dic_m0 <= dic_m6 + 2) "M0_nonspatial" else cmp$model[which.min(cmp$DIC)]

  # ── 3그룹 비공간 M0 적합 (위험요인) ──
  DAT_all <- cm; DAT_urban <- cm %>% filter(area_2=="도시"); DAT_rural <- cm %>% filter(area_2=="농촌")
  res_all   <- run_model(TV_PRESPEC, DAT_all,   quiet=TRUE, re_str=NONSPATIAL, fam=fam)
  res_urban <- run_model(TV_PRESPEC, DAT_urban, quiet=TRUE, re_str=NONSPATIAL, fam=fam)
  res_rural <- run_model(TV_PRESPEC, DAT_rural, quiet=TRUE, re_str=NONSPATIAL, fam=fam)

  out <- list(ok=TRUE, ys=ys, ye=ye, span=ye-ys+1, cm=cm, mor=mor, prep=prep, ic=ic, FMAP=FMAP, cov=cov,
              cmp=cmp, best_nm=best_nm, dic_m0=dic_m0, waic_m0=cmp$WAIC[cmp$model=="M0_nonspatial"],
              res_all=res_all, res_urban=res_urban, res_rural=res_rural,
              N=nrow(ic), cases=sum(cm$cases),
              sig_all=ifelse(is.null(res_all$sig_count),NA,res_all$sig_count),
              sig_urban=ifelse(is.null(res_urban$sig_count),NA,res_urban$sig_count),
              sig_rural=ifelse(is.null(res_rural$sig_count),NA,res_rural$sig_count))

  # Overall 유의 위험요인 방향 지문(안정성 비교용): "kr:부호"
  sig_sig <- character(0)
  if (!is.null(res_all$mv) && nrow(res_all$mv)>0) {
    s <- res_all$mv[res_all$mv$sig=="★", ]
    if (nrow(s)>0) sig_sig <- paste0(s$var_kr, ":", ifelse(s$IRR>1,"+","-"))
  }
  out$sig_signature <- sig_sig

  if (!full) return(out)

  # ── (전체만) 변수별 개별 상호작용 (공변량×도시, 성비·고령 보정) ──
  forced <- intersect(c("sex_ratio_z","elderly_rate_z"), cov)
  int_tbl <- data.frame()
  for (cz in cov) {
    adj <- setdiff(forced, cz); rhs <- paste0(cz, " * urban")
    if (length(adj) > 0) rhs <- paste(rhs, "+", paste(adj, collapse=" + "))
    fs <- paste("cases ~", rhs, "+ offset(log(population+1))", NONSPATIAL)
    fit <- fit_inla(fs, ic, fam=fam); if (is.null(fit)) next
    fe <- fit$summary.fixed
    irn <- grep(paste0("^", cz, ":urban$|^urban:", cz, "$"), rownames(fe), value=TRUE)
    if (!(cz %in% rownames(fe)) || length(irn)!=1) next
    br <- fe[cz,]; bi <- fe[irn,]
    int_tbl <- rbind(int_tbl, data.frame(
      Variable = mk_kr(cz, FMAP),
      IRR_rural = round(exp(br$mean),3), rural_CrI = sprintf("%.3f–%.3f", exp(br$`0.025quant`), exp(br$`0.975quant`)),
      IRR_urban_pt = round(exp(br$mean + bi$mean),3),
      Interaction_IRR = round(exp(bi$mean),3),
      interaction_CrI = sprintf("%.3f–%.3f", exp(bi$`0.025quant`), exp(bi$`0.975quant`)),
      Diff_sig = ifelse(bi$`0.025quant`>0 | bi$`0.975quant`<0, "★", ""),
      DIC = round(fit$dic$dic,1), stringsAsFactors=FALSE))
  }
  out$int_tbl <- int_tbl

  # ── NB vs ZINB 우도 비교 (주모델 선택 근거) — 선택 RE(보통 M0 비공간)에서 ──
  m0_nb   <- fits[[best_nm]]                                              # NB (사다리에서 이미 적합)
  m0_zinb <- fit_inla(paste(base_f, RE_LADDER[[best_nm]]), ic, fam="zeroinflatednbinomial1")
  gp <- function(f){ if(is.null(f)||is.null(f$cpo)) return(NA); round(mean(f$cpo$pit,na.rm=TRUE),3) }
  out$fam_cmp <- data.frame(
    likelihood = c("Negative binomial","Zero-inflated NB (type 1)"),
    DIC  = c(ifelse(is.null(m0_nb),NA,round(m0_nb$dic$dic,2)),  ifelse(is.null(m0_zinb),NA,round(m0_zinb$dic$dic,2))),
    WAIC = c(ifelse(is.null(m0_nb),NA,round(m0_nb$waic$waic,2)), ifelse(is.null(m0_zinb),NA,round(m0_zinb$waic$waic,2))),
    mean_PIT = c(gp(m0_nb), gp(m0_zinb)), stringsAsFactors=FALSE)
  # 주모델 적합치(CPO/PIT·Table 2 우도) = fam(=ZINB)
  out$fit_principal <- if (identical(fam, "zeroinflatednbinomial1")) m0_zinb else m0_nb
  out$principal_family <- fam
  out
}

# ══════════════════════════════════════════════════════════════════════════════
# ① 기간 스윕 (2011–2024 … 2015–2024)
# ══════════════════════════════════════════════════════════════════════════════
cat("\n## ① 기간 스윕 (시작연도 2011→2015, 끝 2024)\n")
SW <- list()
for (ys in SWEEP_STARTS) {
  cat(sprintf("\n──── %d–%d (%d년) ────\n", ys, END_YEAR, END_YEAR-ys+1))
  SW[[as.character(ys)]] <- tryCatch(analyze_period(ys, END_YEAR, full=FALSE),
                                     error=function(e){ cat(sprintf("  ❌ %s\n", e$message)); list(ok=FALSE, ys=ys, ye=END_YEAR) })
}

# 스윕 비교표
sweep_tbl <- do.call(rbind, lapply(SW, function(R) {
  if (is.null(R) || !isTRUE(R$ok)) return(data.frame(start=R$ys, span=END_YEAR-R$ys+1, N=NA, cases=NA,
    preMoranI=NA, preMoranP=NA, selected=NA, DIC_M0=NA, dDIC_M6=NA, WAIC_M0=NA,
    sig_all=NA, sig_urban=NA, sig_rural=NA, signature=NA))
  ddic6 <- R$cmp$dDIC[R$cmp$model=="M6_BYM_RW1_IID"]
  data.frame(start=R$ys, span=R$span, N=R$N, cases=R$cases,
    preMoranI=round(R$mor$I,4), preMoranP=round(R$mor$p,4), selected=R$best_nm,
    DIC_M0=R$dic_m0, dDIC_M6=ddic6, WAIC_M0=R$waic_m0,
    sig_all=R$sig_all, sig_urban=R$sig_urban, sig_rural=R$sig_rural,
    signature=paste(R$sig_signature, collapse=" | "), stringsAsFactors=FALSE)
}))
rownames(sweep_tbl) <- NULL
cat("\n  ▣ 기간 스윕 비교표:\n"); print(sweep_tbl, row.names=FALSE)

# ── 주기간 자동선택: 최장기간 우선 + 안정성 + 적합도 ──
#   안정성 = Overall 유의 위험요인 부호집합(signature)이 '최빈 패턴'과 일치하는가.
#   규칙: (1) 수렴(ok)·M0 선택·preMoran 비유의(비공간 일관) 충족 기간만 후보.
#         (2) 후보 signature의 최빈값(majority)을 '안정 패턴'으로 정의.
#         (3) 안정 패턴과 일치하는 후보 중 최장기간(=start 가장 이른) 채택.
#         (4) 동률·불명확 시 DIC_M0/WAIC_M0 더 낮은 기간.
ok_idx <- which(!is.na(sweep_tbl$N) & sweep_tbl$selected=="M0_nonspatial")
auto_start <- NA
if (length(ok_idx) > 0) {
  cand <- sweep_tbl[ok_idx, ]
  sig_mode <- names(sort(table(cand$signature), decreasing=TRUE))[1]
  stable <- cand[cand$signature==sig_mode, ]
  if (nrow(stable)==0) stable <- cand
  stable <- stable[order(stable$start, stable$DIC_M0), ]   # 최장기간(이른 start) 우선, DIC tiebreak
  auto_start <- stable$start[1]
}
PRINCIPAL_START <- if (!is.na(PRINCIPAL_START_OVERRIDE)) PRINCIPAL_START_OVERRIDE else auto_start
cat(sprintf("\n  ★ 주기간(principal) 선택: %s–%d  (자동=%s, 수동override=%s)\n",
    PRINCIPAL_START, END_YEAR, ifelse(is.na(auto_start),"NA",auto_start),
    ifelse(is.na(PRINCIPAL_START_OVERRIDE),"없음",PRINCIPAL_START_OVERRIDE)))
cat("  ※ 자동선택이 부적절하면 PRINCIPAL_START_OVERRIDE 에 원하는 시작연도를 지정 후 재실행.\n")

# ══════════════════════════════════════════════════════════════════════════════
# ② 주기간 전체 분석 (M0~M6 + 3그룹 IRR + 상호작용 + CPO)
# ══════════════════════════════════════════════════════════════════════════════
cat(sprintf("\n## ② 주기간 전체 분석: %d–%d\n", PRINCIPAL_START, END_YEAR))
P <- analyze_period(PRINCIPAL_START, END_YEAR, full=TRUE, fam=PRINCIPAL_FAMILY)
stopifnot(isTRUE(P$ok))
cat(sprintf("  N=%d | 사례합=%d | 공간선택=%s | 주모델 우도=%s | 사전Moran I=%+.4f p=%.4f\n",
    P$N, P$cases, P$best_nm, P$principal_family, P$mor$I, P$mor$p))
cat("\n  모델비교(M0~M6, NB; 공간성 판정):\n"); print(P$cmp[,c("model","DIC","dDIC","WAIC","postMoranI","postMoranP")], row.names=FALSE)
cat("\n  우도 비교(NB vs ZINB, 선택 RE):\n"); print(P$fam_cmp, row.names=FALSE)
cat("\n  변수별 상호작용(농촌 기준, ★=도농 효과 유의차):\n"); print(P$int_tbl, row.names=FALSE)

# ── CPO/PIT 예측진단 (주모델) ──
cpo_pit <- data.frame(metric=character(), value=character())
if (!is.null(P$fit_principal) && !is.null(P$fit_principal$cpo)) {
  cpo <- P$fit_principal$cpo
  nfail <- sum(cpo$failure > 0, na.rm=TRUE)
  mlik  <- -sum(log(cpo$cpo[cpo$cpo>0]), na.rm=TRUE)   # -sum log CPO (작을수록 좋음)
  mpit  <- mean(cpo$pit, na.rm=TRUE)
  cpo_pit <- data.frame(metric=c("CPO failures","-sum log(CPO)","mean PIT (≈0.5 양호)"),
                        value=c(sprintf("%d / %d", nfail, length(cpo$cpo)), sprintf("%.2f", mlik), sprintf("%.3f", mpit)))
  cat(sprintf("\n  [CPO/PIT] 실패 %d/%d | -Σlog(CPO)=%.2f | 평균 PIT=%.3f\n",
      nfail, length(cpo$cpo), mlik, mpit))
}

# ══════════════════════════════════════════════════════════════════════════════
# ③ 보충 민감도
# ══════════════════════════════════════════════════════════════════════════════
cat("\n## ③ 보충 민감도\n")

# (S-a) 식품매개 2020–2024 (사육두수·어패류 포함) — Overall
CUR_YS <<- SENS_START; CUR_YE <<- SENS_END
cm_food <- build_cor_merged(DZ, SENS_START, SENS_END, include_food=TRUE)
res_food <- tryCatch(run_model(TV_PRESPEC_SUB, cm_food, quiet=TRUE, re_str=NONSPATIAL, fam=PRINCIPAL_FAMILY), error=function(e) NULL)
cat(sprintf("  [S-a 식품매개 2020–2024] 사례=%d | 유의 %s\n", sum(cm_food$cases),
    ifelse(is.null(res_food$sig_count),"NA",res_food$sig_count)))

# (S-b) COVID 연도(2020–2021) 제외 — 주기간 Overall 재적합
ic_noc <- P$ic %>% filter(!year %in% c(2020,2021))
base_f_p <- paste("cases ~", paste(P$cov, collapse=" + "), "+ offset(log(population+1))")
fit_noc <- fit_inla(paste(base_f_p, NONSPATIAL), ic_noc, fam=PRINCIPAL_FAMILY)
covid_tbl <- data.frame()
if (!is.null(fit_noc)) {
  fe <- fit_noc$summary.fixed; fe <- fe[rownames(fe)!="(Intercept)", , drop=FALSE]
  covid_tbl <- data.frame(Variable=sapply(rownames(fe), function(z) mk_kr(z, P$FMAP)),
    IRR=round(exp(fe$mean),3), CrI=sprintf("%.3f–%.3f", exp(fe$`0.025quant`), exp(fe$`0.975quant`)),
    Sig=ifelse(fe$`0.025quant`>0|fe$`0.975quant`<0,"★",""), row.names=NULL)
  cat(sprintf("  [S-b COVID 2020–21 제외] N=%d | 유의 %d\n", nrow(ic_noc), sum(covid_tbl$Sig=="★")))
}

# (S-c) 사례 vs 건수(outbreak) 단위 민감도 — 데이터에 outbreak 컬럼 있을 때만
#   식중독최종.csv 에서 사례수 외 '건수/발생건수/outbreak' 류 컬럼 자동탐지.
outbreak_tbl <- data.frame(); ob_col <- NA
ob_cand <- grep("건수|발생건수|outbreak|건수합|사건", names(DF_RAW_CACHE), value=TRUE, ignore.case=TRUE)
ob_cand <- setdiff(ob_cand, c("cases"))
if (length(ob_cand) >= 1) {
  ob_col <- ob_cand[1]
  cat(sprintf("  [S-c outbreak 단위] 탐지 컬럼='%s' → 건수 기준 재집계\n", ob_col))
  dz_ob <- DF_RAW_CACHE %>% filter(disease==DZ, year>=PRINCIPAL_START, year<=END_YEAR) %>%
    clean_region(PRINCIPAL_START, END_YEAR) %>% group_by(region, year) %>%
    summarise(outb=sum(.data[[ob_col]], na.rm=TRUE), .groups="drop")
  ic_ob <- P$ic %>% left_join(dz_ob, by=c("region","year"))
  ic_ob$cases_bak <- ic_ob$cases; ic_ob$cases <- ifelse(is.na(ic_ob$outb), 0, ic_ob$outb)
  fit_ob <- fit_inla(paste(base_f_p, NONSPATIAL), ic_ob, fam=PRINCIPAL_FAMILY)
  if (!is.null(fit_ob)) {
    fe <- fit_ob$summary.fixed; fe <- fe[rownames(fe)!="(Intercept)", , drop=FALSE]
    outbreak_tbl <- data.frame(Variable=sapply(rownames(fe), function(z) mk_kr(z, P$FMAP)),
      IRR_outbreak=round(exp(fe$mean),3), CrI=sprintf("%.3f–%.3f", exp(fe$`0.025quant`), exp(fe$`0.975quant`)),
      Sig=ifelse(fe$`0.025quant`>0|fe$`0.975quant`<0,"★",""), row.names=NULL)
  }
} else {
  cat("  [S-c outbreak 단위] 식중독최종.csv 에 건수/outbreak 컬럼 없음 → 생략(사례 단위만 보고)\n")
}

# ══════════════════════════════════════════════════════════════════════════════
# ④ 저장 — 본문(4 디스플레이) + 보충자료 분리
#   본문 4개: Table 1(기술), Figure 1(기술/도농), Table 2(IRR), Figure 2(추론)
#   → R은 '표 데이터'를 xlsx로 저장하고, 그림은 BC_main_figures.py 가 이 xlsx들을 읽어 생성.
# ══════════════════════════════════════════════════════════════════════════════
cat("\n## ④ 저장\n")
s_t <- createStyle(fontSize=12, textDecoration="bold")
s_h <- createStyle(textDecoration="bold", halign="center", fgFill="#D9E2F3")
s_s <- createStyle(textDecoration="bold", fontColour="#C00000")

# ── (본문) MAIN workbook ──
wb <- createWorkbook()

# Table 1 데이터: 기술통계(도농별) — 사례/발생률 + 공변량 평균
add_desc <- function(ws_title) {
  addWorksheet(wb, ws_title)
  hdr <- sprintf("%s — Descriptive statistics by urban/rural, %d-%d", EN, PRINCIPAL_START, END_YEAR)
  writeData(wb, ws_title, hdr, startRow=1); addStyle(wb, ws_title, s_t, rows=1, cols=1)
  cm <- P$cm
  grp <- cm %>% mutate(grp=ifelse(area_2=="도시","Urban","Rural")) %>%
    group_by(grp) %>% summarise(region_years=n(), cases=sum(cases),
      incidence_per100k=round(sum(cases)/sum(population)*1e5,3),
      .groups="drop")
  tot <- data.frame(grp="Overall", region_years=nrow(cm), cases=sum(cm$cases),
      incidence_per100k=round(sum(cm$cases)/sum(cm$population)*1e5,3))
  desc <- rbind(tot, as.data.frame(grp))
  writeData(wb, ws_title, desc, startRow=3, colNames=TRUE); addStyle(wb, ws_title, s_h, rows=3, cols=1:ncol(desc), gridExpand=TRUE)
  # 공변량 평균(원자료 단위): FMAP$eng → TV_PRESPEC 의 원(raw) 코드로 역매핑
  raw_code <- TV_PRESPEC$code[match(P$FMAP$eng, TV_PRESPEC$eng)]
  keep <- !is.na(raw_code) & raw_code %in% names(cm)
  if (any(keep)) {
    rc <- raw_code[keep]; eng <- P$FMAP$eng[keep]
    cm2 <- cm %>% mutate(grp=ifelse(area_2=="도시","Urban","Rural"))
    cvO <- sapply(rc, function(v) round(mean(as.numeric(cm[[v]]), na.rm=TRUE),3))
    cvU <- sapply(rc, function(v) round(mean(as.numeric(cm2[[v]][cm2$grp=="Urban"]), na.rm=TRUE),3))
    cvR <- sapply(rc, function(v) round(mean(as.numeric(cm2[[v]][cm2$grp=="Rural"]), na.rm=TRUE),3))
    cvt <- data.frame(Variable=eng, Overall=cvO, Urban=cvU, Rural=cvR, row.names=NULL)
    writeData(wb, ws_title, "Covariate means (raw units)", startRow=3+nrow(desc)+2, startCol=1)
    writeData(wb, ws_title, cvt, startRow=3+nrow(desc)+3, colNames=TRUE)
  }
}
add_desc("Table1_Descriptive")

# Table 2 데이터: 비공간 M0 보정 IRR (Overall/Urban/Rural)
write_grp <- function(ws, res) {
  addWorksheet(wb, ws)
  writeData(wb, ws, sprintf("%s — Non-spatial M0 (region IID + time RW1), zero-inflated NB, adjusted IRR, %s [%d-%d]", EN, ws, PRINCIPAL_START, END_YEAR), startRow=1); addStyle(wb,ws,s_t,rows=1,cols=1)
  if (is.null(res$mv)||nrow(res$mv)==0){ writeData(wb,ws,"No results",startRow=3); return() }
  hd <- c("Category","Variable","Form","IRR","95% CrI Lo","95% CrI Hi","Sig","Observed","Theory","Match")
  writeData(wb,ws,t(hd),startRow=3,startCol=1,colNames=FALSE); addStyle(wb,ws,s_h,rows=3,cols=1:10,gridExpand=TRUE)
  mv <- res$mv
  for (i in 1:nrow(mv)){ r<-3+i; v<-c(mv$카테고리[i],mv$var_kr[i],mv$형태[i],mv$IRR[i],mv$lo[i],mv$hi[i],mv$sig[i],mv$obs_dir[i],mv$theory_dir[i],mv$방향일치[i])
    for (cc in 1:10) writeData(wb,ws,v[cc],startRow=r,startCol=cc); if (mv$sig[i]=="★") addStyle(wb,ws,s_s,rows=r,cols=4:7,gridExpand=TRUE) }
}
write_grp("Table2_Overall", P$res_all); write_grp("Table2_Urban", P$res_urban); write_grp("Table2_Rural", P$res_rural)

# NB vs ZINB 우도 비교 (주모델 선택 근거)
addWorksheet(wb, "NB_vs_ZINB")
writeData(wb,"NB_vs_ZINB", sprintf("%s — Likelihood comparison at non-spatial M0 (principal = zero-inflated NB)", EN), startRow=1); addStyle(wb,"NB_vs_ZINB",s_t,rows=1,cols=1)
writeData(wb,"NB_vs_ZINB","Lower DIC/WAIC = better fit; mean PIT ≈ 0.5 = well-calibrated. ZINB selected on DIC/WAIC.", startRow=2)
if (!is.null(P$fam_cmp)) { writeData(wb,"NB_vs_ZINB", P$fam_cmp, startRow=4, colNames=TRUE); addStyle(wb,"NB_vs_ZINB",s_h,rows=4,cols=1:ncol(P$fam_cmp),gridExpand=TRUE) }

# ── Figure 생성용 데이터 (Python에서 읽음) ──
# (Figure 1 & 2 시각화용: 연간추이, 계절성, 지역별 발생률, 도농 통계)
addWorksheet(wb, "FigData_AnnualTrend")
yrly <- P$cm %>% mutate(UR=ifelse(area_2=="도시","Urban","Rural")) %>%
  group_by(year, UR) %>% summarise(cases=sum(cases), pop=mean(population), .groups="drop") %>%
  mutate(rate=cases/pop*1e5)
writeData(wb,"FigData_AnnualTrend", "Year | Urban/Rural | Cases | Population | Incidence_per_100k", startRow=1)
writeData(wb,"FigData_AnnualTrend", yrly[c("year","UR","cases","pop","rate")], startRow=2, colNames=FALSE)

addWorksheet(wb, "FigData_RegionalRate")
reg_rate <- P$cm %>% group_by(region, area_2) %>% summarise(cases=sum(cases), pop=sum(population), .groups="drop") %>%
  mutate(rate=cases/pop*1e5)
writeData(wb,"FigData_RegionalRate", reg_rate[c("region","area_2","cases","pop","rate")], startRow=1, colNames=TRUE)

addWorksheet(wb, "FigData_Seasonality")
# 월별 사례수 (도농별) - P$cm에서만 추출 (b 데이터 사용 안 함)
tryCatch({
  # P$cm에 month 컬럼이 있다면 사용
  if ("month" %in% names(P$cm)) {
    seasonal <- P$cm %>%
      mutate(UR=ifelse(area_2=="도시","Urban","Rural")) %>%
      filter(!is.na(month)) %>%
      group_by(month, UR) %>%
      summarise(cases=n(), .groups="drop")
    if (nrow(seasonal) > 0) {
      writeData(wb,"FigData_Seasonality", seasonal, startRow=1, colNames=TRUE)
    } else {
      writeData(wb,"FigData_Seasonality", "No seasonal data available", startRow=1)
    }
  } else {
    # month 컬럼 없음 → 더미 메시지
    writeData(wb,"FigData_Seasonality", "Month information not available in data", startRow=1)
  }
}, error = function(e) {
  writeData(wb,"FigData_Seasonality", paste("Seasonality data extraction error:", e$message), startRow=1)
})

addWorksheet(wb, "FigData_Descriptive")
desc_stat <- data.frame(
  Group = c("Overall","Overall","Urban","Urban","Rural","Rural"),
  Statistic = c("Cases","Regions_x_Years","Cases","Regions_x_Years","Cases","Regions_x_Years"),
  Value = c(sum(P$cm$cases), nrow(P$cm), sum(P$cm[P$cm$area_2=="도시","cases"]), nrow(P$cm[P$cm$area_2=="도시",]),
            sum(P$cm[P$cm$area_2=="농촌","cases"]), nrow(P$cm[P$cm$area_2=="농촌",]))
)
writeData(wb,"FigData_Descriptive", desc_stat, startRow=1, colNames=TRUE)

# Figure-source: 모델비교 / 상호작용 (Figure 2 패널 b,c,d 소스)
addWorksheet(wb, "Fig_ModelComparison"); writeData(wb,"Fig_ModelComparison", P$cmp, startRow=1, colNames=TRUE)
addWorksheet(wb, "Fig_Interaction")
writeData(wb,"Fig_Interaction","Per-covariate urban×rural interaction (rural=ref, ★=differs)", startRow=1); addStyle(wb,"Fig_Interaction",s_t,rows=1,cols=1)
if (nrow(P$int_tbl)>0){ writeData(wb,"Fig_Interaction",P$int_tbl,startRow=3,colNames=TRUE); addStyle(wb,"Fig_Interaction",s_h,rows=3,cols=1:ncol(P$int_tbl),gridExpand=TRUE)
  for (i in 1:nrow(P$int_tbl)) if (P$int_tbl$Diff_sig[i]=="★") addStyle(wb,"Fig_Interaction",s_s,rows=3+i,cols=1:ncol(P$int_tbl),gridExpand=TRUE) }
fn_main <- file.path(SUBMIT, sprintf("%s_MAIN_%dto%d_%s.xlsx", SLUG, PRINCIPAL_START, END_YEAR, TS))
saveWorkbook(wb, fn_main, overwrite=TRUE)
cat(sprintf("  ★ 본문 MAIN: %s\n", fn_main))

# ── (보충) SUPP workbook ──
wb2 <- createWorkbook()
addWorksheet(wb2, "S1_PeriodSweep")
writeData(wb2,"S1_PeriodSweep", sprintf("%s — Period sensitivity (start-year sweep, end %d); principal = %d-%d", EN, END_YEAR, PRINCIPAL_START, END_YEAR), startRow=1); addStyle(wb2,"S1_PeriodSweep",s_t,rows=1,cols=1)
writeData(wb2,"S1_PeriodSweep", sweep_tbl, startRow=3, colNames=TRUE); addStyle(wb2,"S1_PeriodSweep",s_h,rows=3,cols=1:ncol(sweep_tbl),gridExpand=TRUE)
addWorksheet(wb2, "S2_ModelComparison_full"); writeData(wb2,"S2_ModelComparison_full", P$cmp, startRow=1, colNames=TRUE)
addWorksheet(wb2, "S3_Interaction_full")
if (nrow(P$int_tbl)>0) writeData(wb2,"S3_Interaction_full", P$int_tbl, startRow=1, colNames=TRUE)
addWorksheet(wb2, "S4_CPO_PIT"); if (nrow(cpo_pit)>0) writeData(wb2,"S4_CPO_PIT", cpo_pit, startRow=1, colNames=TRUE)
addWorksheet(wb2, "S5_COVIDexcluded"); if (nrow(covid_tbl)>0) writeData(wb2,"S5_COVIDexcluded", covid_tbl, startRow=1, colNames=TRUE)
addWorksheet(wb2, "S6_OutbreakUnit")
if (nrow(outbreak_tbl)>0) writeData(wb2,"S6_OutbreakUnit", outbreak_tbl, startRow=1, colNames=TRUE) else writeData(wb2,"S6_OutbreakUnit", sprintf("No outbreak-count column found in 식중독최종.csv (detected candidates: %s)", paste(ob_cand, collapse=", ")), startRow=1)
addWorksheet(wb2, "S7_FoodSensitivity_2020_2024")
if (!is.null(res_food) && !is.null(res_food$mv) && nrow(res_food$mv)>0) {
  writeData(wb2,"S7_FoodSensitivity_2020_2024", sprintf("%s — Food-pathway model (2020-2024, livestock & shellfish), Overall", EN), startRow=1); addStyle(wb2,"S7_FoodSensitivity_2020_2024",s_t,rows=1,cols=1)
  writeData(wb2,"S7_FoodSensitivity_2020_2024", res_food$mv, startRow=3, colNames=TRUE)
}
fn_supp <- file.path(SUBMIT, sprintf("%s_SUPP_%dto%d_%s.xlsx", SLUG, PRINCIPAL_START, END_YEAR, TS))
saveWorkbook(wb2, fn_supp, overwrite=TRUE)
cat(sprintf("  ★ 보충 SUPP: %s\n", fn_supp))

# 그림 생성용 메타(파이썬이 읽음) — jsonlite 있으면 사용, 없으면 수동 작성
meta_path <- file.path(SUBMIT, "BC_run_meta.json")
meta_json <- sprintf('{\n  "disease_en": "%s",\n  "slug": "%s",\n  "principal_start": %d,\n  "end_year": %d,\n  "main_xlsx": "%s",\n  "supp_xlsx": "%s",\n  "ts": "%s"\n}',
  EN, SLUG, PRINCIPAL_START, END_YEAR, basename(fn_main), basename(fn_supp), TS)
writeLines(meta_json, meta_path)

cat(sprintf("\n✅ 완료 | 주기간 %d–%d | 본문 4 디스플레이 + 보충자료\n", PRINCIPAL_START, END_YEAR))
cat("   다음: results/B_cereus/code/BC_main_figures.py 실행 → 본문 Figure 1·2 (멀티패널 a–d) 생성\n")
