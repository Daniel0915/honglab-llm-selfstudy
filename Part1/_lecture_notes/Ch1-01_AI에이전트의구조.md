AI 에이전트의 구조
AI 에이전트의 구조 – 핵심 정리
영상 요약
AI 에이전트
Russell & Norvig의 AI: A Modern Approach에서 AI 연구의 핵심 프레임워크로 다뤄지며 "환경을 인식하고 행동하는 존재(perceiving and acting)"로 정의됩니다.
창발적 능력
LLM은 처음에는 다음 토큰(단어)을 예측하도록 훈련되었지만 모델 규모가 커지고 더 다양한 데이터를 사용하는 과정에서 Q&A, 코딩, 요약 등 의도하지 않은 능력들이 나타나기 시작했습니다. 이를 창발적 능력(emergent ability)이라고 부릅니다.
무상태(stateless)
LLM 자체는 기억이 없습니다. 기억처럼 보이는 것은 알고보면 내부적으로 매번 과거의 대화나 필요한 정보를 반복해서 넣어주는 것입니다.
LLM은 어떻게 에이전트가 되었나
자연어 소통	LLM이 인간의 언어를 이해	GPT-3 (2020)
↓	
	

추론/계획 + 도구 사용	AI 에이전트의 기본 구조	CoT (2022), ReAct (2022)
Toolformer (2023)
↓	
	

도구 생성	직접 코딩하여 새 도구 생성	Voyager (2023)
↓	
	

자기발전 루프	실행 → 평가 → 개선 반복	Self-Refine (2023)
Reflexion (2023)
↓	
	

기억 관리	경험을 저장하고 자체적으로 관리	MemGPT (2023)
참고 자료
AI: A Modern Approach (Russell & Norvig) — AI 에이전트 정의
GPT-3 (Brown et al., 2020) — 별도의 추가 훈련 없이 프롬프트 안의 예시만으로 새로운 작업을 수행하는 few-shot learning 능력을 보여주며, 이후 in-context learning이라 불리게 됨
Emergent Abilities of LLMs (Wei et al., 2022) — 모델 규모 증가에 따른 다수의 창발적 능력 확인
Chain-of-Thought Prompting (Wei et al., 2022) — 단계별 추론 능력
ReAct (Yao et al., 2022) — 추론과 도구 사용의 결합
Toolformer (Schick et al., 2023) — LLM이 자체적으로 도구 사용법을 학습
Voyager (Wang et al., 2023) — 마인크래프트 에이전트가 새로운 스킬 코드를 자동 생성하고 축적
Self-Refine (Madaan et al., 2023) — 자체 피드백을 통한 반복 개선
Reflexion (Shinn et al., 2023) — 언어 기반 자기발전 루프
MemGPT (Packer et al., 2023) — LLM의 계층적 기억 관리
MARK INCOMPLETE
