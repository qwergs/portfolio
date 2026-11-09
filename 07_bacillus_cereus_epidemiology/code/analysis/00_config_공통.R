# ==============================================================================
# 00_config_공통.R — FBD 도시/농촌 공간분석 (전 질병 통합) 공통 설정
#
# 설계(확정):
#   - 질병을 인자로 받아 루프 도는 단일 일반화 파이프라인
#   - 공통 변수 풀 + AUTO (전 질병 동일 절차 → 비교 일관성)
#   - 메인 기간 2011–2024 (장기 커버 공변량만; 다년 선형보간 + 커버리지 게이트로
#     2011을 못 덮는 후발시작 변수[1인가구 2015~·손씻기 격년·식품 2019~]는 자동 제외)
#   - 민감도 기간 2020–2024 (식품매개 변수: 사육두수·어패류 포함)
#   - 보조: 사전 Moran 공간성 분류는 공변량 무관 → 2011–2024 장기로 안정적
#   - 모델: 도시/농촌 3그룹(전체·도시·농촌), INLA M6 BYM+RW1+IID, NB
#   - 공간상관 판단: 사전 Moran's I, p<0.05 = 공간상관 있음
# ==============================================================================
suppressPackageStartupMessages({
  library(dplyr); library(tidyr); library(MASS); library(stringr); library(car)
  library(openxlsx); library(arrow); library(sf); library(spdep); library(INLA)
})
options(scipen = 999)

# ── INLA 안정화·재현성 (중요) ──
#   num.threads="1:1" : 멀티스레드 부동소수점 비결정성 제거 → 재실행 시 동일 결과(일관성)
#   inla.mode="compact": 기본 모드 유지(고정효과 정상 추정). classic은 이 모델에서 β를 0으로 죽임 → 사용 안 함.
#   ※ vb.correction 경고는 무해(추정 정상). NAN/INF rescue도 INLA 자가복구.
suppressWarnings(try(INLA::inla.setOption(num.threads = "1:1", inla.mode = "compact"), silent = TRUE))

# ── 경로 ──
BASE_IV  <- "/Users/guns/Desktop/건상_시각화/FBD_DATA_ZIP"          # 데이터(읽기)
DIR_RES  <- "/Users/guns/Desktop/도시_농촌_분석/results"            # 결과(쓰기)
DIR_LOG  <- "/Users/guns/Desktop/도시_농촌_분석/logs"               # 로그(쓰기)
PATH_DISEASE   <- file.path(BASE_IV, "식중독최종.csv")
PATH_HEALTH_PQ <- file.path(BASE_IV, "국민건강결과_최종.parquet")
PATH_SHP       <- file.path(BASE_IV, "final.shp")
for (d in c(DIR_RES, DIR_LOG)) if (!dir.exists(d)) dir.create(d, recursive = TRUE)

# ── 분석 질병 (건수충분 12종; 필요시 보류5종 추가) ──
DISEASES_MAIN <- c(
  "노로바이러스","장염비브리오","살모넬라","병원성대장균","황색포도상구균",
  "캠필로박터제주니","클로스트리디움퍼프린젠스","바실러스세레우스",
  "장출혈성대장균감염증","세균성이질","장티푸스","비브리오패혈증")
DISEASE_EN <- c(
  "노로바이러스"="Norovirus","장염비브리오"="V. parahaemolyticus",
  "살모넬라"="Salmonella","병원성대장균"="Pathogenic E. coli",
  "황색포도상구균"="S. aureus","캠필로박터제주니"="C. jejuni",
  "클로스트리디움퍼프린젠스"="C. perfringens","바실러스세레우스"="B. cereus",
  "장출혈성대장균감염증"="EHEC","세균성이질"="Shigellosis",
  "장티푸스"="Typhoid fever","비브리오패혈증"="V. vulnificus",
  "콜레라"="Cholera","파라티푸스"="Paratyphoid","원충"="Protozoa",
  "A형간염"="Hepatitis A","E형간염"="Hepatitis E")
disease_slug <- function(d) {  # 결과 폴더/파일명용 안전 슬러그
  s <- DISEASE_EN[d]; if (is.na(s)) s <- d
  gsub("[^A-Za-z0-9]+","_", s)
}

# ── 기간 설정 ──
MAIN_START <- 2011; MAIN_END <- 2024          # 메인 (장기)
SENS_START <- 2020; SENS_END <- 2024          # 식품변수 민감도

# ── 결측 보간 + 스팬 게이트 파라미터 ──
# 스팬 게이트: 실측이 [ys+MAX_CARRY, ye-MAX_CARRY] 양끝을 못 덮으면 메인 제외.
#   2011–2024·MAX_CARRY=2 → 실측이 2013 전 시작 & 2022 후 종료해야 통과.
#   (1인가구 2015~·식품 2019~ 자동 드롭, 손씻기 격년은 내부 선형보간으로 통과)
MAX_CARRY <- 2   # 내부 선형보간 후 양끝 결측은 인접 실측값을 최대 2년까지만 연장(거짓 역추정 방지)

# ── 분석 파라미터 (NORO v8 동일) ──
PVAL_SCREEN   <- 0.20
VIF_THRESHOLD <- 10
MIN_OBS       <- 20
COV_RATIO     <- 0.85
HIGH_RISK_PCT <- 0.20
FAMILY        <- "nbinomial"
TARGET_FWD_ALL <- 5; TARGET_FWD_URBAN <- 5; TARGET_FWD_RURAL <- 5
MAX_ITER       <- 400

# ── 공통 강제 보정변수(인구학적) ──
FORCED_VARS <- c("성비","고령인구비율")

# ── 건강지표(parquet) 공통 변수 ──
HEALTH_VARS <- c("1인가구수.1","1인가구수_65세이상","1인가구율_45-64세가구","1인가구율_65세이상가구","1인가구율_전체",
  "가정의학과전문의","건강생활실천율_조율","걷기실천율_표준화율","격렬한신체활동실천율_표준화율","고령인구비율",
  "고위험음주율_남_표준화","고위험음주율_여_표준화","관내진료비_외래","관내진료비_입원","관내진료비_전체",
  "관내진료실인원_외래","관내진료실인원_입원","관내진료실인원_전체","관외진료비_외래","관외진료비_입원","관외진료비_전체",
  "관외진료실인원_외래","관외진료실인원_입원","관외진료실인원_전체","국민기초생활보장수급자","국민기초생활보장수급자수율",
  "기준시간내접근불가비율_종합병원(전체)","기초생활수급자수율","기초연금수급자수","내과전문의",
  "노인장기요양_시설_기관수","노인장기요양시설_영양사","농촌인구수","다문화이혼건수","다문화이혼비중","다문화출생비율",
  "다문화출생아수","다문화혼인비중","도시인구수","도시지역면적","도시지역인구비율","독거노인가구비율","독거노인비율",
  "목욕시설_있음","방역수칙실천율실내마스크착용_표준화율","방역수칙실천율실외마스크착용_표준화율",
  "보건및사회복지사업체수","보건및사회복지사업체종사자비율","보건및사회복지사업체종사자수","보건소인력_보건직",
  "비누,손세정제사용률_표준화율","비누손세정제사용률_표준화율","상수도보급률","성비","수도_마을상수도","수도_상수도",
  "수도_전용상수도","순이동인구","식사전손씻기실천율_표준화율","식품안정성확보율_표준화율",
  "어제저녁식사후칫솔질실천율_표준화율","어제점심식사후칫솔질실천율_표준화율","연간인플루엔자예방접종률_표준화율",
  "예방의학과전문의","온수시설_있음","외출후손씻기실천율_표준화율","우울감경험률_표준화율","월간음주율_남_표준화",
  "월간음주율_여_표준화","의사수","의원_가정의학과","의원_예방의학과","의원_재활의학과","인구천명당사설학원수",
  "인구천명당의료기관종사의사수","인구천명당주점업수","인구천명당패스트푸드점수","작업치료사수","재정자립도","재정자주도",
  "재활의학과전문의","전문의합계","정신건강의학과전문의","주관적건강수준인지율_표준화율","주점업수",
  "중등도이상신체활동실천율_조율","집밖에서의손소독제사용횟수_표준화율","총가구수_65세이상","치과의사수",
  "코로나19관련유증상자행동수칙미준수율_표준화율","패스트푸드점수","평소손씻기실천율_표준화율",
  "하수도보급률","화장실_수세식","화장실_재래식","화장실다녀온후손씻기실천율_조율")

# ── 일반 CSV 공통 변수 (지정 로드) ──
SELECTED <- list(
  "merged_세부용도별지하수.csv"=c("온천수_시설수","온천수_이용량"),
  "merged_지하수수질.csv"=c("검사합계","부적합","적합","합계"),
  "merged_하수관로개보수.csv"=c("개·보수관로_부분보수(개소)_계","개·보수관로_전체보수(m)_합류식","맨홀(개소)_합류식맨홀","받이_빗물받이(개소)","토실,토구_토실(개소)"),
  "merged_하수도보급률.csv"=c("고도처리인구보급률(%)","공공하수처리구역인구보급률(%)","총면적(㎢)","총인구(명)","하수도설치율(%)","하수처리구역내_계","하수처리구역외_계","하수처리구역외_오수처리인구","하수처리구역외_정화조인구"),
  "국토이용현황_전처리_수정.csv"=c("공장용지","구거","답","대","도로","목장용지","묘지","유지","임야","잡종지","전","제방","창고용지","하천"),
  "생활용지하수이용현황_전처리.csv"=c("가정용_개소수","가정용_이용량","간이상수도용_개소수","간이상수도용_이용량","민방위용_개소수","민방위용_이용량","일반용_개소수","일반용_이용량","총 계_개소수","총 계_이용량","학교용_개소수","학교용_이용량"),
  "음용지하수이용현황_전처리.csv"=c("개소수(총합)","민방위용","총이용량"),
  "재정자립도_전처리.csv"=c("재정자립도(세입과목개편전)","재정자립도(세입과목개편후)"),
  "추가_1인가구_전처리.csv"=c("1인가구"),
  "추가_독거노인가구비율_전처리.csv"=c("65세이상_1인가구(가구)","독거노인가구비율(%)"),
  "손씻기_전처리.csv"=c("after_outing_handwash_rate_adj","after_outing_handwash_rate_std","after_toilet_handwash_rate_adj","before_meal_handwash_rate_adj","usual_handwash_rate_adj","usual_handwash_rate_std","비누_손_세정제_사용률_표준화율"),
  "merged_분뇨찌꺼기처리.csv"=c("발생량(A)=(B)+(C)","처분량_계(B)","처분량_재활용"),
  "1인가구_전처리.csv"=c("주택_계","주택_다세대주택","주택_단독주택","주택_아파트","주택_연립주택")
)
SELECTED_NEW <- list(
  "교육기관_인력_예산.csv"=c("교원1인당학생수","유치원교원수","초등학교교원수","유치원원아수",
    "초등학교 학생 수","유치원 수","유아천명당보육시설수","초등학교 수",
    "전문대학 및 대학교 수","인구천명당 사설학원수"),
  "경제수준.csv"=c("주택소유율","청년고용률_상반기","청년고용률_하반기",
    "기초연금수급자율","국민기초생활보장수급자수율","국민연금_임의가입자수")
)

# ── 식품매개/후반시작 파일 (메인 제외, 2020–2024 민감도에서만 로드) ──
FOOD_FILES_SELECTED <- list(
  "가축두수_전처리.csv"=c("농가수(호)_가금","농가수(호)_돼지","농가수(호)_젖소","농가수(호)_한육우","농가수(호)_합계",
    "사육두수(두)_가금","사육두수(두)_돼지","사육두수(두)_젖소","사육두수(두)_한육우","사육두수(두)_합계")
)
# 어패류/양식/의료시설/종합소득세/하수찌꺼기/고령인구는 02_load에서 별도 처리(카테고리 합산 등)

# 메타/텍스트 제외 컬럼
META_COLS <- c("region","year","sido","sigungu","X","Unnamed..0","항목",
  "분뇨처리장명","타시설연계여부","시설명","소재지","처리공법","연계처리장명",
  "전화번호","가동개시일","운영방법","방류수역_지류","방류수역_본류","방류수역_수계",
  "연계처리","연계명","행정코드","시설코드","지역.시.군.구.","하수처리장명","연도")

# ══════════════════════════════════════════
# 공통 함수 (NORO v8 동일)
# ══════════════════════════════════════════
clean_region <- function(df, ys, ye) df %>% mutate(
  region = str_replace_all(as.character(region), "\\s+", ""),
  region = if_else(region == "인천시미추홀구", "인천시남구", region),
  year = as.integer(year)) %>% filter(year >= ys, year <= ye)

read_csv_safe <- function(fp) { raw <- NULL
  for (enc in c("UTF-8","UTF-8-BOM","CP949","EUC-KR")) {
    raw <- tryCatch(read.csv(fp, fileEncoding = enc, check.names = FALSE, stringsAsFactors = FALSE), error = function(e) NULL)
    if (!is.null(raw) && nrow(raw) > 0) break }; raw }

# 인접 1년 보간 (양끝)
fill_missing_year <- function(df, tgt, src) {
  if (!"region" %in% names(df) || !"year" %in% names(df)) return(df)
  nv <- setdiff(names(df)[sapply(df, is.numeric)], "year")
  if (length(nv) == 0 || !src %in% unique(df$year)) return(df)
  ds <- df %>% filter(year == src); dt <- df %>% filter(year == tgt)
  df_f <- ds %>% mutate(year = as.integer(tgt))
  if (nrow(dt) > 0) df_f <- df_f %>%
    left_join(dt %>% dplyr::select(region, all_of(nv)) %>% rename_with(~paste0(., "__o"), all_of(nv)), by = "region") %>%
    mutate(across(all_of(nv), function(col){ v <- cur_column(); o <- get(paste0(v, "__o")); ifelse(!is.na(o), o, col) })) %>%
    dplyr::select(region, year, all_of(nv))
  bind_rows(df %>% filter(year != tgt), df_f) %>% arrange(region, year)
}
# 기간에 맞춰 양끝 결측연도만 보간 (메인 2015-2024 / 민감도 2020-2024 공통)
apply_cf <- function(df, ye) fill_missing_year(df, ye, ye - 1)

is_pct <- function(x){ xv <- x[!is.na(x) & is.finite(x)]; all(xv >= 0 & xv <= 100) & max(xv) > 1 }

run_univ <- function(x, df_w) {
  tmp <- data.frame(cases = df_w$cases, x = x, pop = df_w$population)
  tmp <- tmp[complete.cases(tmp) & is.finite(tmp$x) & tmp$pop > 0, ]
  if (nrow(tmp) < MIN_OBS || sd(tmp$x, na.rm = TRUE) == 0) return(NULL)
  tryCatch({ m <- glm.nb(cases ~ x + offset(log(pop + 1)), data = tmp); cr <- summary(m)$coefficients
    if (nrow(cr) < 2) return(NULL)
    list(p = cr[2,"Pr(>|z|)"], IRR = exp(cr[2,"Estimate"]),
      lo = exp(cr[2,"Estimate"] - 1.96*cr[2,"Std. Error"]),
      hi = exp(cr[2,"Estimate"] + 1.96*cr[2,"Std. Error"]), n = nrow(tmp))
  }, error = function(e) NULL)
}

# ── 도시/농촌 분류 ──
metro_pattern <- "^(서울시|부산시|대구시|인천시|광주시|대전시|울산시)"
classify_region <- function(r) {
  if (grepl("세종", r)) return("대도시")
  if (grepl(metro_pattern, r) & !grepl("군$", r)) return("대도시")
  if (grepl("군$", r)) return("농촌")
  "중소도시"
}

# ── 이론방향/카테고리 자동배정 (FBD 공통, NORO v8 동일) ──
assign_theory_dir <- function(v) {
  if (grepl("사육두수|농가수|가축|가금|돼지|한육우|젖소|말$", v)) return("위험")
  if (grepl("부적합|오염|폐수|배출|유기물질", v)) return("위험")
  if (grepl("지하수|민방위용|일반용|학교용|가정용|간이상수도|총이용량|개소수\\(총합\\)", v)) return("위험")
  if (grepl("주점|패스트푸드|음주", v)) return("위험")
  if (grepl("미접속|정화조|하수처리구역외", v)) return("위험")
  if (grepl("독거노인|65세이상.*1인|1인가구.*65", v)) return("위험")
  if (grepl("기초생활|수급자", v)) return("위험")
  if (grepl("미준수|우울", v)) return("위험")
  if (grepl("화장실_재래", v)) return("위험")
  if (grepl("외부위탁|하수찌꺼기|자체처리|함수율|탈수", v)) return("위험")
  if (grepl("맨홀|토실", v)) return("위험")
  if (grepl("농촌인구", v)) return("위험")
  if (grepl("유지|구거|하천|답$|제방|목장용지", v)) return("위험")
  if (grepl("손씻기|세정|소독|칫솔|마스크|거리두기|방역", v)) return("보호")
  if (grepl("식품안정", v)) return("보호")
  if (grepl("의사|의원|전문의|의료|병원|보건소|보건직|작업치료", v)) return("보호")
  if (grepl("예방접종|건강생활|걷기|신체활동", v)) return("보호")
  if (grepl("하수도.*보급|하수.*처리.*내|고도처리|상수도|수세식|공공하수|하수도설치", v)) return("보호")
  if (grepl("국민연금|재정자립|재정자주", v)) return("보호")
  if (grepl("온수시설|목욕시설|보수관로|개·보수|전체보수", v)) return("보호")
  if (grepl("검사합계|적합$", v)) return("보호")
  if (grepl("장기요양.*기관|영양사|보건.*사회복지", v)) return("보호")
  if (grepl("진료비|진료실인원|기초연금", v)) return("보호")
  "중립"
}
assign_category <- function(v) {
  if (grepl("사육두수|농가수|가축|가금|돼지|한육우|젖소|말$", v)) return("① 병원소 및 축산")
  if (grepl("자체처리|소각|건조|외부위탁|하수찌꺼기|함수율|탈수|고화", v)) return("② 분뇨 및 오염처리")
  if (grepl("폐수|유기물질|오염|부적합|적합$|검사합계|하수도|정화조|미접속|하수처리|공공하수|상수도|맨홀|토실|보수관로", v)) return("③ 수질 및 하수 인프라")
  if (grepl("용지|토지|답$|전$|임야|하천|유지|구거|묘지|잡종|광천|도로|대$|제방|주유소|창고|목장", v)) return("④ 수계 및 토지이용")
  if (grepl("손씻기|세정|소독|칫솔|마스크|거리두기|방역|위생|화장실|식품안정|미준수", v)) return("⑤ 위생 및 식품안전")
  if (grepl("재정|국민연금|1인가구|순이동|다문화|주점|패스트|기초생활|수급자|독거|농촌", v)) return("⑥ 사회경제 및 취약성")
  if (grepl("고령|성비|인구비율", v)) return("⑦ 인구학적 특성")
  if (grepl("의사|의원|전문의|의료|병원|보건소|예방접종|건강|우울|음주|진료|기초연금", v)) return("⑧ 의료접근")
  if (grepl("지하수|온천수|민방위|일반용|학교용|가정용|간이상수도|총이용량|개소수\\(총합\\)", v)) return("③ 수질 및 하수 인프라")
  "⑨ 기타"
}

# ══════════════════════════════════════════
# 사전지정(PRE-SPECIFIED) 공통 이론 변수셋 — 교수님 방식 (AUTO 제거)
#   · 2011–2024에서 스팬게이트를 통과하는(장기 실측) 변수만으로 구성
#   · 전체/도시/농촌 3그룹 공통 적용(각 그룹 run_model 내 VIF<10이 공선성 자동 정리)
#   · forced(‡) = 인구학적 보정변수(성비·고령인구비율)
#   ※ 기간 한계: ① 축산/식품원(어패류·사육두수 2020~)·⑤ 위생행태(손씻기·화장실=설문/격년)
#     ·② 분뇨는 2011–2024 메인에 실측이 없어 제외됨. 식품원은 2020–2024 민감도에서 확인.
#   ※ 교수님의 질병별 사전지정 셋이 확정되면 이 표를 질병별로 교체.
TV_PRESPEC <- data.frame(
  tier = "A",
  cat = c("⑦ 인구학적 특성","⑦ 인구학적 특성","⑦ 인구학적 특성",
          "③ 수질 및 하수 인프라","③ 수질 및 하수 인프라","③ 수질 및 하수 인프라",
          "③ 수질 및 하수 인프라","③ 수질 및 하수 인프라","③ 수질 및 하수 인프라",
          "④ 수계 및 토지이용","④ 수계 및 토지이용","④ 수계 및 토지이용",
          "④ 수계 및 토지이용","④ 수계 및 토지이용",
          "⑧ 의료접근","⑧ 의료접근","⑧ 의료접근","⑧ 의료접근","⑧ 의료접근",
          "⑥ 사회경제 및 취약성","⑥ 사회경제 및 취약성"),
  kr = c("성비","고령인구비율","도시지역인구비율",
         "상수도보급률","하수도보급률","고도처리인구보급률",
         "정화조인구","가정용지하수개소수","민방위용지하수이용량",
         "유지","하천","목장용지","답","임야",
         "의사수","관내진료비전체","인플루엔자예방접종률","건강생활실천율","걷기실천율",
         "재정자립도","재정자주도"),
  code = c("성비","고령인구비율","도시지역인구비율",
           "상수도보급률","하수도보급률","고도처리인구보급률(%)",
           "하수처리구역외_정화조인구","가정용_개소수","민방위용_이용량",
           "유지","하천","목장용지","답","임야",
           "의사수","관내진료비_전체","연간인플루엔자예방접종률_표준화율","건강생활실천율_조율","걷기실천율_표준화율",
           "재정자립도","재정자주도"),
  eng = c("sex_ratio","elderly_rate","urban_pop_rate",
          "water_supply","sewerage_rate","adv_treat_rate",
          "septic_pop","gw_home_count","gw_civil_use",
          "reservoir","river","ranch","paddy","forest",
          "physicians","med_total","flu_vax","health_practice","walking_rate",
          "fiscal_indep","fiscal_auto"),
  forced = c("‡","‡", rep("", 19)),
  이론방향 = c("중립","중립","위험",
             "보호","보호","보호",
             "위험","위험","위험",
             "위험","위험","위험","위험","중립",
             "보호","보호","보호","보호","보호",
             "보호","보호"),
  stringsAsFactors = FALSE)

# ── 서브(2020–2024) 전용 변수셋 = 베이스 + 식품매개·축산 (2020~만 존재) ──
#   베이스(2011–2024)에선 스팬게이트로 빠지는 ① 식품원/축산 경로를, 데이터가 있는
#   2020–2024 서브에서 검정. build_cor_merged(include_food=TRUE)가 아래 변수를 적재함.
TV_PRESPEC_SUB <- rbind(
  TV_PRESPEC,
  data.frame(
    tier = "A",
    cat  = c("① 병원소 및 축산","① 병원소 및 축산","① 식품원 및 축산","① 식품원 및 축산"),
    kr   = c("사육두수합계","농가수합계","패류생산총합","갑각류생산총합"),
    code = c("사육두수(두)_합계","농가수(호)_합계","패류_전체총합","갑각류_전체총합"),
    eng  = c("livestock_total","farm_total","shellfish_total","crustacean_total"),
    forced = c("","","",""),
    이론방향 = c("위험","위험","위험","위험"),
    stringsAsFactors = FALSE))

cat(sprintf("✅ 00_config_공통.R 로드 완료 (PRE-SPECIFIED 베이스 %d개 / 서브 %d개)\n",
            nrow(TV_PRESPEC), nrow(TV_PRESPEC_SUB)))
