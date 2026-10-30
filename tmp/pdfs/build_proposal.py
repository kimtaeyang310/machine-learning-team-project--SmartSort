"""기획서 PDF 생성 도구. 직접 실행할 때만 폰트 등록과 파일 생성을 수행합니다."""
from pathlib import Path
import csv, math
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor, Color
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle
from PIL import Image


def build_proposal():
    """캔버스와 페이지 상태를 작업 내부에 두고 페이지별 렌더링 함수를 차례로 실행합니다."""
    ROOT=Path(__file__).resolve().parents[2]
    OUT=ROOT/'output/pdf/SmartSort_AI_Platform_Proposal.pdf'
    OUT.parent.mkdir(parents=True,exist_ok=True)
    pdfmetrics.registerFont(TTFont('KR','C:/Windows/Fonts/malgun.ttf'))
    pdfmetrics.registerFont(TTFont('KRB','C:/Windows/Fonts/malgunbd.ttf'))
    W,H=960,600
    INK='#13352D'; GREEN='#137C60'; LIME='#D8EF9F'; BG='#F5F7F1'; GRAY='#587067'; WHITE='#FFFFFF'; GOLD='#D99A31'
    c=canvas.Canvas(str(OUT),pagesize=(W,H),pageCompression=1)
    c.setTitle('SmartSort | 농산물 숙도 판별과 실행을 연결하는 AI 에이전트 플랫폼')
    c.setAuthor('SmartSort | 김태양 · 류채연')
    page=0
    def box(x,y,w,h,fill=WHITE,r=14,stroke=None):
     c.setFillColor(HexColor(fill));c.setStrokeColor(HexColor(stroke or fill));c.roundRect(x,H-y-h,w,h,r,fill=1,stroke=bool(stroke))
    def txt(x,y,s,size=12,color=INK,bold=False):
     c.setFillColor(HexColor(color));c.setFont('KRB' if bold else 'KR',size);c.drawString(x,H-y-size,s)
    def para(x,y,w,s,size=13,color=INK,bold=False,leading=None):
     st=ParagraphStyle('p',fontName='KRB' if bold else 'KR',fontSize=size,leading=leading or size*1.65,textColor=HexColor(color),wordWrap='CJK')
     p=Paragraph(s,st);_,h=p.wrap(w,1000);p.drawOn(c,x,H-y-h);return h
    def base(k,title,sub=''):
     nonlocal page
     page+=1; c.setFillColor(HexColor(BG));c.rect(0,0,W,H,fill=1,stroke=0)
     txt(40,24,'SMARTSORT',12,GREEN,True);txt(706,26,'AI PLATFORM BUILD CHALLENGE',9,GRAY)
     txt(40,72,k,10,GREEN,True);txt(40,96,title,28,INK,True)
     if sub: para(40,141,880,sub,12,GRAY)
     c.setStrokeColor(HexColor('#D8E0D5'));c.line(40,44,920,44)
     txt(40,566,'SmartSort  |  서비스 기획서  |  2026.10.21',9,GRAY);txt(855,566,f'{page:02d} / 12',9,GRAY)
    def end():c.showPage()
    def card(x,y,w,h,n,title,body):
     box(x,y,w,h);txt(x+20,y+18,n,11,GREEN,True);txt(x+20,y+47,title,18,INK,True);para(x+20,y+86,w-40,body,12)
    def note(s):para(42,527,875,s,9,GRAY)

    def render_page_01():
        base('01 / VISION','잘 익은 판단이, 더 나은 유통으로.','사진 한 장의 숙도 판별을 과수원의 선별·출하 작업으로 연결합니다.')
        box(40,194,548,307,INK)
        txt(65,216,'SmartSort',43,LIME,True)
        para(65,288,490,'농산물 숙도 판별과<br/>선별·출하 우선순위를 연결하는<br/><b>AI 에이전트 플랫폼</b>',24,WHITE,leading=36)
        txt(65,455,'TEAM  김태양 · 류채연',12,LIME)
        for y,n,t,b in [(194,'SEE','숙도를 읽다','바나나·귤·토마토 / 3단계 분류'),(300,'PLAN','행동을 제안하다','수확일·수량·출하 계획을 함께 해석'),(406,'ACT','승인 후 기록하다','작업 목록·담당자·처리 이력 연결')]:
         box(610,y,310,94);txt(630,y+13,n,10,GREEN,True);txt(630,y+33,t,18,INK,True);txt(630,y+65,b,11,GRAY)
        note('본 기획서는 완성형 서비스 경험을 제안합니다. 숙도 분류 기반은 구현되어 있으며, 웹 서비스와 에이전트는 확장 구현 범위입니다.')
        end()

    def render_page_02():
        base('02 / PROBLEM','문제는 판별 이후에 남아 있습니다.','초기 고객: 과수원 운영자와 수확·선별 담당자. 수확 후 숙도 확인과 선별·출하 판단을 돕습니다.')
        card(40,198,280,242,'01','판단의 편차','같은 농산물도 담당자의 경험과 근무 시간에 따라 숙도 판단이 달라집니다.<br/><br/><b>필요한 것</b><br/>공통된 숙도 기준과 확인 가능한 기록')
        card(340,198,280,242,'02','행동으로 이어지지 않음','숙도를 확인해도 어떤 수확물을 먼저 선별하고 출하할지 다시 판단해야 합니다.<br/><br/><b>필요한 것</b><br/>수량·수확일을 반영한 작업 우선순위')
        card(640,198,280,242,'03','개선 근거가 부족함','숙도 판정, 선별 작업, 실제 출하 결과가 연결되지 않으면 운영 방식을 개선하기 어렵습니다.<br/><br/><b>필요한 것</b><br/>판정부터 처리까지 이어지는 이력')
        box(40,460,880,54,LIME);para(58,473,844,'핵심 가설  |  숙도 판별과 작업 제안을 한 흐름으로 묶으면 반복 검수 시간과 출하 준비 지연을 줄일 수 있습니다.',13,bold=True)
        note('문제와 고객 수요는 기획 가설입니다. 현장 인터뷰와 파일럿으로 빈도·중요도·지불 의사를 검증합니다.')
        end()

    def render_page_03():
        base('03 / SOLUTION','촬영부터 오늘의 작업 목록까지, 한 번에.','직원이 과일 종류를 선택하고 촬영하면, 분류 모델과 에이전트가 서로 다른 역할로 협업합니다.')
        steps=[('1','촬영·입력','흰 배경의 과일 1개 촬영<br/>과일 종류·수확일·수량 입력'),('2','숙도 판별','과일별 최적 모델 호출<br/>덜 익음 / 적숙 / 과숙'),('3','실행안 생성','재고 정보와 운영 규칙 조회<br/>우선 선별·출하 검토 제안'),('4','승인·기록','직원이 제안 확인·수정<br/>작업 등록과 처리 결과 저장')]
        for i,(n,t,b) in enumerate(steps):card(40+i*224,207,208,231,n,t,b)
        box(40,462,880,54,INK);para(60,474,840,'사용자 가치  |  “이 과일은 얼마나 익었나?”에서 “지금 무엇부터 할까?”까지 연결합니다.',16,WHITE,True)
        note('초기에는 수확한 과일 1개를 흰 배경 작업대에서 촬영합니다. 나무 위 과실 판별·수확 시점 예측은 후속 검증 범위입니다.')
        end()

    def render_page_04():
        base('04 / PRODUCT EXPERIENCE','과수원 작업을 연결하는 AI 작업 보드','완성형 화면 설계 예시 | 화면의 수량·신뢰도·대화는 설명용 가상 데이터입니다.')
        box(40,195,880,317,WHITE,stroke='#D8E0D5')
        box(40,195,166,317,INK);txt(58,217,'SmartSort',20,LIME,True)
        for y,t in [(278,'오늘의 작업'),(321,'숙도 분석'),(364,'재고 현황'),(407,'처리 이력')]:txt(60,y,t,13,WHITE)
        txt(228,215,'오늘의 확인 우선순위',20,INK,True);txt(228,249,'AI가 제안하고, 과수원 담당자가 결정합니다.',11,GRAY)
        for x,big,label in [(228,'03','확인할 재고'),(454,'02','승인 대기'),(680,'01','재촬영 필요')]:
         box(x,279,204,72,BG);txt(x+14,286,big,25,GREEN,True);txt(x+72,308,label,11,GRAY)
        txt(230,373,'품목 / 상태',10,GRAY);txt(438,373,'AI 제안',10,GRAY);txt(745,373,'작업',10,GRAY)
        for y,item,action,button in [(400,'귤 A / 적숙','우선 출하 후보로 검토','제안 승인'),(447,'토마토 B / 판정 보류','조명을 바꿔 다시 촬영','재촬영')]:
         txt(230,y,item,12,INK,True);txt(438,y,action,12);box(743,y-3,131,31,LIME);txt(760,y+3,button,11,INK,True)
        note('설계 원칙: 모델 결과와 제안 이유를 함께 표시하고, 사진만으로 식품의 안전성·섭취 가능 여부를 확정하지 않습니다.')
        end()

    def render_page_05():
        base('05 / AI AGENT','에이전트는 설명을 넘어, 도구로 작업합니다.','LLM은 이미지 판정을 임의로 만들지 않고, 분류 도구의 구조화된 결과를 받아 실행안을 구성합니다.')
        card(40,197,278,272,'INPUT','상태를 읽고','사용자: “오늘 먼저 확인할 과일을 정리해줘.”<br/><br/>입력: 사진·과일 종류·수확일·수량·농장 운영 규칙<br/><br/>부족한 정보는 질문하고, 인식 실패는 재촬영으로 돌려보냅니다.')
        card(340,197,280,272,'TOOLS','근거를 조회하고','classify_ripeness<br/>숙도·모델 점수·버전 반환<br/><br/>get_inventory / retrieve_policy<br/>재고·농장 승인 운영 규칙 조회<br/><br/>draft_tasks<br/>근거가 포함된 작업 초안 생성')
        card(642,197,278,272,'ACTION','승인 후 실행합니다','출력: 우선순위·권장 행동·판단 근거·추가 확인 항목<br/><br/>confirm_task<br/>담당자 승인 후 작업 저장<br/><br/>동일 요청 중복 등록 방지와 도구 호출 이력을 관리합니다.')
        box(40,484,880,36,LIME);txt(56,491,'예시 응답: “수확일과 적숙 판정을 근거로 귤 A의 출하 검토를 제안합니다. 작업으로 등록할까요?”',12,INK,True)
        note('에이전트·도구 API는 구현 계획입니다. 낮은 확신의 보류 기준은 검증 데이터로 조정하며 모델 점수를 보정 확률로 단정하지 않습니다.')
        end()

    def render_page_06():
        base('06 / TECHNOLOGY','검증된 분류 기반 위에 서비스 계층을 더합니다.','현재 구현과 본선 확장을 분리해, 짧은 개발 기간에도 끝까지 동작하는 데모를 만듭니다.')
        box(40,194,880,105,INK);txt(60,211,'구현 완료  |  숙도 분류 엔진',17,LIME,True)
        para(60,247,840,'이미지 입력 → OpenCV·K-means 전처리 → HSV·색상 비율 특징 → 과일별 분류기<br/>또는 원본 이미지 → YOLOv8n-cls → 숙도·모델 점수 출력',13,WHITE)
        for x,t,b in [(40,'사용자 서비스','모바일 대응 웹 화면<br/>촬영·업로드·작업 보드'),(340,'에이전트·API','Python API + LLM 도구 호출<br/>JSON 스키마·실패 복구'),(640,'운영 데이터','재고·판정·승인 이력 DB<br/>모델 버전·도구 호출 로그')]:
         box(x,323,280,128);txt(x+18,340,t,18,INK,True);para(x+18,378,244,b,12)
        box(40,470,880,46,LIME);txt(58,482,'연결 방식  |  웹 입력 → API 검증 → 분류 엔진 → 에이전트 초안 → 직원 승인 → DB 기록',13,INK,True)
        note('확장 설계: 웹·API·DB·LLM 연계. 배포 전 데이터셋·라이브러리·모델 가중치의 이용 조건과 서비스 적합성을 확인합니다.')
        end()

    def render_page_07():
        base('07 / EVIDENCE','이미 동작하는 기반, 수치로 확인한 출발점','저장된 최신 평가 기록: 2026.10.20 17:13 KST | 3개 숙도 클래스에 대한 평가')
        for x,v,l in [(40,'0.874','전체 Macro F1'),(340,'87.35%','전체 Accuracy'),(640,'모델 6종 비교','과일 3종별 모델 6종 비교')]:
         box(x,194,280,96,INK if x==40 else WHITE);txt(x+20,205,v,26 if x==640 else 32,LIME if x==40 else GREEN,True);txt(x+20,256,l,12,WHITE if x==40 else GRAY)
        rows=[('바나나','Random Forest',.9354,.9048),('귤','YOLOv8n-cls',.9511,.8843),('토마토','YOLOv8n-cls',.7021,.7775)]
        txt(48,313,'과일 / 선택 모델',11,GRAY);txt(356,313,'학습 CV Macro F1',11,GRAY);txt(592,313,'Test Macro F1',11,GRAY)
        for i,(f,m,cv,te) in enumerate(rows):
         y=347+i*49;txt(48,y,f+'  /  '+m,13,INK,True);txt(370,y,f'{cv:.3f}',15,GREEN,True);box(590,y+3,200,14,'#E3E9DE',r=5);box(590,y+3,200*te,14,GREEN,r=5);txt(811,y,f'{te:.3f}',14,INK,True)
        para(42,502,875,'검증 해석: 전체 점수는 과일별 F1의 단순 평균이 아닙니다. 토마토 성능은 개선 여지가 있으며, 현재 평가는 실제 과수원 작업 환경의 성능을 보장하지 않습니다.',10,GRAY)
        note('근거: outputs/evaluation/evaluation_history.csv · outputs/training/by_fruit/model_comparison.csv. CV는 3-fold 그룹 분할 평균이며 Test와 다릅니다.')
        end()

    def render_page_08():
        base('08 / DATA & VALIDATION','현장에 강해지기 위한 검증 설계','기존 학습 체계를 유지하면서, 새 개체·새 조명에서의 신뢰도를 확인합니다.')
        card(40,195,280,263,'DATA','출처와 라벨','귤·토마토: 팀 직접 촬영, 개체·날짜·숙도 라벨 관리<br/><br/>바나나: 외부 이미지 데이터 활용, 같은 원본의 증강 이미지를 그룹화<br/><br/>출력 범위: 덜 익음·적숙·과숙. 품질 등급은 예측하지 않습니다.')
        card(340,195,280,263,'VALIDATION','누수를 줄이는 비교','6개 후보를 과일별 동일한 3-fold 그룹 분할로 비교합니다.<br/><br/>기존 모델의 표준화는 fold 학습 내부에 적용합니다.<br/><br/>YOLO의 조기 종료용 내부 검증과 외부 평가를 구분합니다.')
        card(640,195,280,263,'NEXT','실사용 환경 검증','다양한 과일 개체와 조명·배경·촬영 기기에서 성능을 검증하고, 토마토의 숙도 경계 사례를 보완합니다.<br/><br/>새로운 과일 개체로 성능 확인<br/>촬영 환경별 결과 비교<br/>토마토 숙도 경계 사례 재검수')
        box(40,477,880,42,LIME);para(56,483,848,'안정적인 사용 지원  |  분석이 어려운 사진에는 재촬영을 안내하고, 판단이 애매한 결과는 임의로 판단하지 않고 담당자 확인으로 연결합니다.',12,INK,True,leading=16)
        note('과숙은 부패·안전 판정과 동일하지 않습니다. 내부 품질·당도·잔여 유통기한은 현재 이미지 모델의 예측 대상이 아닙니다.')
        end()

    def render_page_09():
        base('09 / BUSINESS','과수원에서 시작해, 선별·출하 운영 도구로','초기 도입 단위는 과수원 1곳. 스마트폰으로 수확물을 촬영해 선별·출하 작업을 돕는 구독형 서비스입니다.')
        card(40,197,280,253,'CUSTOMER','누가 사용하나','과수원 운영자와 수확·선별 담당자<br/><br/>우선 공략: 현재 지원 품목 중 귤 재배 농가의 수확 후 선별 작업<br/><br/>확장: 공동선별장·농산물 산지유통센터')
        card(340,197,280,253,'VALUE','왜 계속 사용하나','단순 사진 분류: 숙도 결과 제공<br/><br/>SmartSort 제안: 숙도 결과에 재고 정보·운영 규칙을 연결하고 승인 작업과 처리 결과를 축적<br/><br/>선별 기준과 작업 인수인계의 일관성 강화')
        card(640,197,280,253,'REVENUE','어떻게 수익화하나','파일럿: 제한된 품목과 촬영량으로 효용 검증<br/><br/>기본 구독: 숙도 분석·작업 보드<br/><br/>운영 구독: 작업장 관리·리포트<br/>확장 API: 기존 재고 시스템 연동')
        box(40,470,880,47,INK);para(58,482,845,'가격 결정 기준  |  농가당 선별 확인 시간 절감 가치와 추론·저장 비용을 측정한 뒤, 인터뷰로 지불 의사를 검증합니다.',12,WHITE)
        note('시장 규모·유료 고객·폐기 감소 실적은 아직 검증되지 않았습니다. 사업 모델과 기대 효과는 파일럿 검증 대상입니다.')
        end()

    def render_page_10():
        base('10 / DELIVERY PLAN','분류 엔진에서, 시연 가능한 서비스까지','예선 기획 이후 본선 제출을 향한 실행 계획 | 대회 공식 안내의 최종 산출물 제출: 2026.11.25 23:00')
        for i,(n,t,b) in enumerate([('W1','분류 API·데이터 정리','입력·응답 스키마 고정<br/>모델 버전과 실패 응답 표준화'),('W2','웹·에이전트 연결','촬영 화면과 재고 입력<br/>도구 호출·작업 초안 생성'),('W3','승인·예외 처리','승인 저장·중복 방지<br/>보류·재촬영·로그 확인'),('W4','통합 검증·시연','미사용 이미지 검증<br/>3분 데모·설치 문서 완성')]):card(40+224*i,200,208,196,n,t,b)
        box(40,416,430,100,WHITE);txt(58,431,'김태양 | 데이터·서비스 기획',16,INK,True);para(58,466,392,'라벨 검수·사용자 인터뷰·화면 흐름·시연 구성',12)
        box(490,416,430,100,WHITE);txt(508,431,'류채연 | 모델·서비스 구현',16,INK,True);para(508,466,392,'모델 평가·분류 API·에이전트 연계·통합 검증',12)
        note('팀 역할은 저장소의 기존 데이터/모델 담당을 기반으로 확장 제안했습니다. 세부 일정은 대회 공지와 팀 가용 시간에 맞춰 조정합니다.')
        end()

    def render_page_11():
        base('11 / SUCCESS CRITERIA','기능 완성과 현장 효용을 함께 검증합니다.','아래 수치는 향후 개발·파일럿의 목표이며, 이미 달성한 실적이 아닙니다.')
        targets=[('분류 품질','과일별 Macro F1 ≥ 0.85','미사용 개체 단위 Test로 측정. 전체 점수와 과일별 점수를 함께 보고합니다.'),('처리 속도','숙도 분류 속도 17 FPS 이상','초당 이미지 17장 처리 목표. 지정 CPU에서 전처리·분류 시간을 측정하며, 업로드·에이전트 응답은 제외합니다.'),('에이전트 동작','에이전트 작업 완료율 90% 이상','30개 이상의 시험 상황에서 정보 조회·작업 제안·담당자 승인 후 저장까지 올바르게 수행하는지 평가합니다.'),('현장 효용','숙도 확인 작업 시간 단축','기존 수작업과 서비스 사용 시 전체 작업 시간을 비교해 시간 절감 효과를 검증합니다.')]
        for i,(a,b,d) in enumerate(targets):
         y=191+i*78;box(40,y,880,66,WHITE);txt(57,y+14,a,12,GREEN,True);txt(194,y+10,b,17,INK,True);para(194,y+36,696,d,10,GRAY)
        note('본선 시연: 정상 분석 → 재고 기반 제안 → 직원 승인 → 작업 이력 확인 → 불확실 입력의 재촬영까지 한 흐름으로 보여줍니다.')
        end()

    def render_page_12():
        base('12 / SCOPE & REFERENCES','실행 가능한 기반에서, 완성도 있는 서비스로','현재 구현 내용과 서비스로 확장할 수 있는 예시를 구분해 제시합니다.')
        box(40,189,424,133,INK);txt(58,204,'현재 구현',18,LIME,True);para(58,242,388,'3개 과일·3단계 숙도 분류, OpenCV 특징 추출, 6개 모델 비교, 과일별 모델 선택, 폴더 예측, 평가 이력과 특징 캐시',12,WHITE)
        box(484,189,436,133,WHITE);txt(502,204,'확장 가능한 서비스 예시',18,GREEN,True);para(502,242,400,'모바일 웹, 재고 DB, LLM 에이전트·도구 API, 승인 기반 작업 보드 등으로 확장할 수 있습니다. 제시한 화면과 대화는 구현 가능성을 보여주는 예시입니다.',12)
        txt(42,341,'자료 출처와 사용 범위',15,INK,True)
        refs=[('프로젝트 근거','README.md · docs/performance_history.md · src/ · 최신 학습 비교 및 평가 CSV'),('대회 안내','daker.ai/public/hackathons/ai-platform-build-challenge-prelims (2026.10.21 확인)'),('외부 데이터','Mendeley Data: data.mendeley.com/datasets/ptfscwtnyz/2 / Kaggle: leftin/fruit-ripeness-unripe-ripe-and-rotten'),('참고 데이터 목록','Kaggle: sumn2u/riped-and-unriped-tomato-dataset / Roboflow Universe (README 기재 자료)'),('오픈소스·도구','Python · OpenCV · NumPy · pandas · scikit-learn · joblib · Ultralytics YOLOv8n-cls'),('작성 지원','AI 도구(Codex): 저장소 분석, 서비스 기획 정리, 화면 시안·PDF 편집 지원')]
        for i,(a,b) in enumerate(refs):
         y=373+i*24;txt(43,y,a,9,GREEN,True);para(160,y,755,b,9,GRAY,leading=13)
        note('외부 데이터의 실제 학습 포함 범위·개별 라이선스는 배포 전 재점검합니다. 현장 도입·매출·폐기 절감 효과를 달성 실적으로 주장하지 않습니다.')
        end()

    render_page_01()
    render_page_02()
    render_page_03()
    render_page_04()
    render_page_05()
    render_page_06()
    render_page_07()
    render_page_08()
    render_page_09()
    render_page_10()
    render_page_11()
    render_page_12()
    c.save()
    print(OUT)
    print(f'pages={page}, bytes={OUT.stat().st_size}, MB={OUT.stat().st_size/1_000_000:.3f}')


if __name__ == "__main__":
    build_proposal()
