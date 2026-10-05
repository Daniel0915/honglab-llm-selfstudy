[안내] 무료 코딩 AI
실습은 제가 AI와 함께 만들어가는 과정을 보여드리는 방식입니다.
아직 코딩 AI 사용 경험이 없으신 분들을 위해서 무료 서비스 안내해드립니다.
저는 클로드로 시범을 보여드리지만 제미나이나 챗GPT로 실습을 끝내신 분들도 계십니다.
1. 완전 무료
도구	특징	비고
Google Antigravity	제미나이 · 클로드 등 여러 모델 선택 · 주간 사용량 제한	신용카드 불필요 · VSCode와 별개 프로그램
GitHub Copilot Free	자동완성 월 2,000회 · 채팅 월 50회	모델을 고를 수 없고 작은 모델이 배정됨
ChatGPT Free	GPT-5 채팅 + Codex 에이전트	5시간 단위로 사용량 제한
코파일럿은 VSCode 마켓플레이스에서 확장을 설치하고 깃허브 계정으로 로그인합니다. Antigravity는 별도 프로그램을 내려받아 설치하며, 터미널에서 쓰는 agy 명령만 따로 설치할 수도 있습니다(윈도우 파워셸에서 irm https://antigravity.google/cli/install.ps1 | iex). 이렇게 하면 VSCode 안의 터미널에서 그대로 쓸 수 있습니다.
Gemini Code Assist 확장은 2026년 6월 18일부터 개인 계정의 요청을 받지 않습니다. 구글이 Antigravity로 통합했고, 개인용 제미나이 CLI도 같은 날 함께 종료되었습니다.
2. 유료
플랜	월 가격	특징
Google AI Pro	$20 (약 28,000원)	Antigravity 사용량 확대
Claude Pro	$20 (약 28,000원)	Claude Code 포함 (VSCode 확장 지원)
Claude Max	$100 (약 141,000원)부터	Pro의 5배 · 20배 사용량
ChatGPT Plus	$20 (약 28,000원)	Codex 포함
기타 자신에게 편한 다른 AI 서비스를 사용하셔도 전혀 상관 없습니다.
3. 윈도우에서 빌드가 안 될 때
AI가 코드는 잘 써놓고 "터미널을 직접 열어서 실행해달라"며 빌드를 못 끝내는 경우가 있습니다. 컴파일러 경로가 잡혀 있지 않아서입니다.
윈도우에서 C를 컴파일하려면 Visual Studio 2022 Community가 필요합니다. Visual Studio Installer에서 "C++를 사용한 데스크톱 개발"을 선택해 설치합니다.
설치해도 일반 터미널에서 cl이 바로 실행되지는 않습니다. 같은 세션에서 vcvars64.bat을 먼저 실행해야 경로가 잡힙니다.
AI에게 두 명령을 따로 시키면 세션이 이어지지 않아 실패합니다. 한 개의 .bat 파일로 묶어서 실행하게 하면 됩니다.
@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
chcp 65001 >nul
cl /nologo hello.c /Fe:hello.exe
hello.exe
시작 메뉴에서 "Developer Command Prompt for VS 2022"를 열고 그 안에서 cl hello.c를 실행해도 됩니다. 이 창은 컴파일러 경로가 미리 잡혀 있습니다. chcp 65001은 한글이 깨지지 않게 하는 설정입니다.
무료 플랜은 작은 모델이 배정되기 때문에 이런 환경 문제를 스스로 해결하지 못하고 사용자에게 넘기는 일이 잦습니다. 위 .bat 파일을 만들어두고 AI에게 이 파일을 실행하라고 알려주시면 됩니다.
