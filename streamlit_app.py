# streamlit_app.py
# 실행: streamlit run --server.port 3000 --server.address 0.0.0.0 streamlit_app.py

import io
import datetime
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns  # 테마만 사용 (원하면 제거 가능)
import streamlit as st

# -----------------------------
# ✅ 한국어 폰트 강제 등록 (절대경로)
# -----------------------------
import matplotlib
from matplotlib import font_manager as fm, rcParams
from pathlib import Path

# Pretendard-Bold.ttf 절대 경로 지정
font_path = Path("fonts/Pretendard-Bold.ttf").resolve()

if font_path.exists():
    fm.fontManager.addfont(str(font_path))
    font_prop = fm.FontProperties(fname=str(font_path))
    rcParams["font.family"] = font_prop.get_name()  # 전역 기본 폰트명으로도 설정
    print("✅ Loaded font:", font_prop.get_name())
else:
    font_prop = fm.FontProperties()  # fallback
    print("⚠️ 폰트를 찾을 수 없음:", font_path)

rcParams["axes.unicode_minus"] = False  # 마이너스 기호 깨짐 방지

# -----------------------------
# Streamlit 설정
# -----------------------------
st.set_page_config(layout="wide", page_title="CO₂ & Global Temperature Dashboard")
st.title("🌍 대기 중 CO₂ 농도와 지구 평균 기온, 무슨 관계가 있을까?")
st.caption("데이터 출처: NOAA GML(마우나로아 CO₂), NASA GISTEMP(지구 평균 기온 이상치)")

# -----------------------------
# 안전한 텍스트 페치 유틸
# -----------------------------
def fetch_text(url: str, timeout: int = 12) -> list[str]:
    """간단한 UA/타임아웃을 가진 텍스트 로더"""
    req = Request(url, headers={"User-Agent": "Mozilla/5.0 (Streamlit classroom app)"})
    with urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8").splitlines()

# -----------------------------
# 데이터 로더
# -----------------------------
@st.cache_data(show_spinner=False)
def load_co2_mlo_monthly_to_annual() -> pd.DataFrame:
    """
    NOAA GML Mauna Loa 월별 CO₂ 텍스트를 로드하여 연평균으로 변환
    URL: https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.txt
    출력: DataFrame[year:int, co2_ppm:float]
    """
    url = "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.txt"
    lines = fetch_text(url)

    rows = []
    for line in lines:
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            year = int(parts[0])
            month = int(parts[1])
            val = float(parts[3])  # average 열
        except Exception:
            continue
        rows.append([year, month, val])

    dfm = pd.DataFrame(rows, columns=["year", "month", "co2_ppm"])
    dfa = dfm.groupby("year", as_index=False)["co2_ppm"].mean()
    dfa = dfa.dropna().sort_values("year").reset_index(drop=True)
    return dfa


@st.cache_data(show_spinner=False)
def load_global_temp_anomaly_annual(start_year=1880, end_year=None) -> pd.DataFrame:
    """
    NASA GISTEMP Global Land+Ocean 연평균 기온 이상치 로드
    URL: https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv
    - 헤더(Year) 라인 탐지 후 CSV 파싱
    - 'J-D'(연평균) 사용
    - 문자열→숫자 변환, 센티-섭씨(×100) 여부 자동 판별
    출력: DataFrame[Year:int, TempAnomaly:float(℃)]
    """
    if end_year is None:
        end_year = datetime.date.today().year

    url = "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv"
    txt = fetch_text(url)

    # 헤더 라인 탐지
    header_idx = next(i for i, line in enumerate(txt) if line.strip().startswith("Year"))
    df = pd.read_csv(io.StringIO("\n".join(txt[header_idx:])))

    target_col = "J-D" if "J-D" in df.columns else df.columns[1]
    out = df[["Year", target_col]].rename(columns={target_col: "TempAnomaly"}).dropna()

    # 문자열 → 숫자
    out["TempAnomaly"] = pd.to_numeric(out["TempAnomaly"], errors="coerce")
    out = out.dropna()

    # 스케일 자동 판별(센티℃ → ℃)
    if out["TempAnomaly"].abs().median() > 5:
        out["TempAnomaly"] = out["TempAnomaly"] / 100.0

    # 연도 범위 필터
    out = out[(out["Year"] >= start_year) & (out["Year"] <= end_year)].copy()
    out["Year"] = out["Year"].astype(int)
    out = out.sort_values("Year").reset_index(drop=True)
    return out

# -----------------------------
# 데이터 로드
# -----------------------------
with st.spinner("데이터 로딩 중... 잠시만 기다려 주세요! 🚀"):
    try:
        co2_annual = load_co2_mlo_monthly_to_annual()
        temp_annual = load_global_temp_anomaly_annual(1880, datetime.date.today().year)
    except Exception as e:
        st.error(f"데이터를 불러오는 중 문제가 발생했습니다: {e}")
        st.stop()

# 공통 연도 계산(슬라이더 범위)
yr_min = int(max(co2_annual["year"].min(), temp_annual["Year"].min()))
yr_max = int(min(co2_annual["year"].max(), temp_annual["Year"].max()))

st.sidebar.header("연도 범위 선택")
yr_start, yr_end = st.sidebar.slider(
    "보고 싶은 기간을 골라보세요!", min_value=yr_min, max_value=yr_max,
    value=(max(1960, yr_min), yr_max), step=1
)
smooth = st.sidebar.checkbox("12년 이동평균 (전체적인 흐름 보기)", value=True)

# -----------------------------
# 데이터 결합
# -----------------------------
co2_r = co2_annual[(co2_annual["year"] >= yr_start) & (co2_annual["year"] <= yr_end)].copy()
tmp_r = temp_annual[(temp_annual["Year"] >= yr_start) & (temp_annual["Year"] <= yr_end)].copy()
df = pd.merge(co2_r.rename(columns={"year": "Year"}), tmp_r, on="Year", how="inner")

# 빈 데이터 가드
if df.empty or len(df) < 2:
    st.warning("선택한 연도 범위에 공통 데이터가 부족합니다. 슬라이더 범위를 넓혀 보세요.")
    st.stop()

# 스무딩
if smooth and len(df) >= 12:
    df["co2_ppm_smooth"] = df["co2_ppm"].rolling(12, center=True, min_periods=1).mean()
    df["TempAnomaly_smooth"] = df["TempAnomaly"].rolling(12, center=True, min_periods=1).mean()

# 최신 연도/적용 범위 캡션
st.caption(
    f"적용 연도 범위: {int(df['Year'].min())}–{int(df['Year'].max())} "
    f"(GISTEMP 최신 연도: {int(temp_annual['Year'].max())}, CO₂ 최신 연도: {int(co2_annual['year'].max())})"
)

# -----------------------------
# 시각화
# -----------------------------
st.subheader("📈 CO₂ 농도와 지구 평균 기온, 같이 볼까요?")

sns.set_theme(style="whitegrid")
fig, ax1 = plt.subplots(figsize=(10.5, 5.2))

# CO₂ (좌축)
ax1.plot(df["Year"], df["co2_ppm"], lw=1.6, color="#1f77b4", alpha=0.45, label="CO₂ 농도 (연평균)")
if smooth and "co2_ppm_smooth" in df.columns:
    ax1.plot(df["Year"], df["co2_ppm_smooth"], lw=2.8, color="#1f77b4", label="CO₂ 농도 (장기 추세)")
ax1.set_xlabel("연도", fontproperties=font_prop)
ax1.set_ylabel("대기 중 CO₂ (ppm)", color="#1f77b4", fontproperties=font_prop)
ax1.tick_params(axis="y", labelcolor="#1f77b4")

# 기온 이상치 (우축)
ax2 = ax1.twinx()
ax2.plot(df["Year"], df["TempAnomaly"], lw=1.6, color="#d62728", alpha=0.45, label="기온 변화 (연평균)")
if smooth and "TempAnomaly_smooth" in df.columns:
    ax2.plot(df["Year"], df["TempAnomaly_smooth"], lw=2.8, color="#d62728", label="기온 변화 (장기 추세)")
ax2.set_ylabel("지구 평균 기온 변화 (℃)", color="#d62728", fontproperties=font_prop)
ax2.tick_params(axis="y", labelcolor="#d62728")

# 제목(한글 폰트 강제)
plt.title(f"CO₂ 농도와 지구 평균 기온 변화 ({yr_start}–{yr_end})",
          pad=10, fontproperties=font_prop)

# 범례 통합
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=False, prop=font_prop)

fig.tight_layout()
st.pyplot(fig, clear_figure=True)

# -----------------------------
# 요약 지표
# -----------------------------
c1, c2, c3 = st.columns(3)
c1.metric("CO₂ 얼마나 늘었을까?", f"{df['co2_ppm'].iloc[-1] - df['co2_ppm'].iloc[0]:+.1f} ppm")
c2.metric("기온은 얼마나 변했을까?", f"{df['TempAnomaly'].iloc[-1] - df['TempAnomaly'].iloc[0]:+.2f} ℃")
c3.metric("얼마나 관련 있을까? (상관계수)", f"{np.corrcoef(df['co2_ppm'], df['TempAnomaly'])[0,1]:.2f}")

with st.expander("데이터 표로 확인하기"):
    st.dataframe(
        df[["Year", "co2_ppm", "TempAnomaly"]].rename(
            columns={"Year": "연도", "co2_ppm": "CO₂(ppm)", "TempAnomaly": "기온 변화(℃)"}
        ),
        use_container_width=True
    )

# 병합 데이터 다운로드
csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    "📥 분석용 CSV 내려받기 (병합본)",
    data=csv_bytes,
    file_name=f"co2_temp_merged_{yr_start}_{yr_end}.csv",
    mime="text/csv"
)

# -----------------------------
# 📘 데이터 해석 (모둠 관점)
# -----------------------------
st.markdown("---")
st.header("📘 데이터 탐구 보고서: 우리 모둠의 발견")

st.subheader("1. 대기 중 CO₂ 농도의 지속적인 증가")
st.markdown("""
그래프의 파란색 선(CO₂ 농도)을 보면 알 수 있듯이, CO₂ 농도가 지속적으로 상승하는 모습은 저희 모둠에게 상당히 인상적이었습니다. 
저희가 태어나기 전인 1960년대 약 320ppm에서 현재 420ppm을 초과하는 수치에 도달한 것을 확인했습니다. 
이는 단순히 숫자의 변화를 넘어, 인류의 활동이 지구 대기 환경 전체에 영향을 미치고 있다는 명확한 증거라고 생각되어 책임감을 느끼게 되었습니다.
""")

st.subheader("2. '기온 이상치' 상승의 의미")
st.markdown("""
빨간색 선(기온 변화)으로 표시된 '기온 이상치'는 특정 기준값과의 차이를 의미합니다. NASA에서는 **1951년부터 1980년까지의 30년 평균 기온**을 그 기준으로 사용합니다. 
즉, 그래프의 0℃ 선이 바로 이 기간의 평균 기온이며, 각 연도의 값은 이 기준보다 얼마나 기온이 높았는지(플러스 값) 또는 낮았는지(마이너스 값)를 보여주는 것입니다.

분석 결과, 최근에는 기준치보다 매년 0.5℃ 이상 높았으며, 근래에는 1℃를 초과하는 해도 관측되었습니다. 
1℃라는 수치가 작게 느껴질 수 있지만, 이것이 전 지구적인 폭염, 폭우 등 극단적 기상 현상의 원인이 된다는 사실을 배우며 문제의 심각성을 체감할 수 있었습니다.
""")

st.subheader("3. CO₂ 농도와 기온 변화의 뚜렷한 상관관계")
st.markdown("""
이번 탐구에서 가장 주목할 만한 점은 **파란색 CO₂ 농도 선과 빨간색 기온 변화 선이** 매우 유사한 형태로 함께 상승한다는 사실이었습니다. 
CO₂ 농도가 증가함에 따라 기온 역시 상승하는 뚜렷한 경향성을 발견했습니다. 
이는 과학 시간에 배운 온실효과를 데이터로 직접 확인하는 과정이었으며, 눈에 보이지 않는 기체가 지구 전체의 온도를 높여 우리 삶에 직접적인 영향을 미칠 수 있다는 사실을 실감하게 했습니다.
""")

st.subheader("4. 탐구를 통해 느낀 점")
st.markdown("""
이번 프로젝트는 단순한 과제를 넘어, 데이터를 통해 미래 사회의 문제를 읽어내는 의미 있는 경험이었습니다. 
기후 위기가 막연한 미래의 일이 아닌, 우리가 살고 있는 현재의 문제임을 데이터를 통해 명확히 인식하게 되었습니다. 
이에 저희 모둠은 앞으로 교실 소등, 분리배출과 같은 일상 속 작은 실천부터 책임감을 갖고 행동하기로 다짐했습니다.
""")

# -----------------------------
# 📢 우리 세대를 위한 제언
# -----------------------------
st.markdown("---")
st.header("📢 우리 세대를 위한 제언")

st.markdown("""
저희는 이번 프로젝트를 통해 기후 위기가 교과서 속 지식이 아닌, 우리 모두의 현실임을 확인했습니다. 
따라서 같은 시대를 살아가는 학생들에게 다음과 같이 제안하고자 합니다.
""")

st.markdown("""
**1. 작은 실천의 중요성** 일상 속에서 무심코 사용하는 에너지를 절약하고, 급식 잔반을 남기지 않고, 일회용품 사용을 줄이는 등의 작은 습관이 모여 큰 변화를 만들 수 있습니다.

**2. 데이터 기반의 소통** "지구가 아프다"는 감성적인 호소와 더불어, 객관적인 데이터를 근거로 토론하고 소통할 때 더 큰 설득력을 가질 수 있습니다.

**3. 학교 공동체 내에서의 활동** 환경 동아리 활동이나 학급 캠페인을 통해 기후 위기 문제에 대한 공감대를 형성하고, 학교 차원의 해결 방안을 함께 고민해 볼 수 있습니다.

**4. 미래 진로와의 연계** 기후 위기 문제를 해결하기 위한 과학 기술, 사회 정책 등 관련 분야로의 진로를 탐색하는 것은 우리 세대가 미래를 준비하는 또 다른 방법이 될 것입니다.

기후 위기는 거대하고 어려운 문제이지만, 데이터를 통해 현상을 정확히 이해하고 함께 행동한다면 충분히 해결해 나갈 수 있습니다. 
우리 세대의 관심과 실천이 지속 가능한 미래를 만드는 첫걸음이 될 것이라고 믿습니다. 🌱
""")

# -----------------------------
# 📚 참고자료
# -----------------------------
st.markdown("---")
st.header("📚 참고자료")

st.markdown("""
- **데이터 출처**
    - [NOAA Global Monitoring Laboratory - Mauna Loa CO₂ Data](https://gml.noaa.gov/ccgg/trends/data.html)
    - [NASA GISS Surface Temperature Analysis (GISTEMP v4)](https://data.giss.nasa.gov/gistemp/)
- **추천 도서**
    - 그레타 툰베리, 《기후 책》, 이순희 역, 기후변화행동연구소 감수, 열린책들, 2023. 
      ([Yes24 도서 정보 링크](https://www.yes24.com/product/goods/119700330))
""")

# -----------------------------
# Footer (팀명)
# -----------------------------
st.markdown(
    """
    <div style='text-align: center; padding: 20px; color: gray; font-size: 0.9em;'>
        미림마이스터고 1학년 4반 2조 · 지구야아프지말아조
    </div>
    """,
    unsafe_allow_html=True
)
