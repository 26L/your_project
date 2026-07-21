# 사용 프로필 — 범용 vs 커뮤니티(조직 특화)

> 벤치마크 결론은 **"하나의 승자"가 아니라 용도별 두 프로필**이다(§10.7~10.9).
> 질문이 연결(다중홉·종합)을 실제로 요구하는지, 문서가 진짜 연결돼 있는지로 갈린다.

## 한눈에

| | **범용 프로필** | **커뮤니티(조직 특화) 프로필** |
| --- | --- | --- |
| 추천 기법 | `hybrid` (벡터+BM25 RRF) | `graphrag_e2b_hybrid` (그래프+벡터) / global 많으면 `graphrag_e2b_adaptive` |
| 언제 | 일반 QA, 핀포인트·다중홉, 문서 연결 약함 | 특정 조직/커뮤니티의 **연결된** 문서, 종합·global·연결성 질의 |
| 그래프 추출 | 불필요 | 필요(비용·시간↑) |
| 속도 | 최고속(~1.3s) | 느림(~3s, 추출 별도) |
| config | `config/profile_general.yaml` | `config/profile_community.yaml` |

## 프로필 A — 범용

**언제**: 문서 간 연결이 약하거나, 질문이 대부분 특정 사실/2홉이면. 대부분의 실무가 여기 해당.

**근거**
- 내부 매트릭스 judge **0.806(공동 1위)**, single/multi/relational 최고, **최고속**.
- 외부 HotpotQA(다중홉 n=100)에서도 평면과 **통계적 동률**(McNemar 무유의차).
- 그래프 추출 비용 0, 도메인 로버스트.

**실행**
```bash
ragbench index --method hybrid --config config/profile_general.yaml --data <corpus>
ragbench eval  --method hybrid --config config/profile_general.yaml --eval-set <set> --judge
```

## 프로필 B — 커뮤니티(조직 특화)

**언제**: 한 조직/커뮤니티의 문서가 서로 **실제로 참조·연결**되고, "전사적으로/각 부서가/어떻게 이어지나" 같은 **종합·global·연결성** 질문이 중요할 때.

**근거**
- **global 유형에서 그래프 계열이 역전**(dynamic/e2b/adaptive 0.375 > 평면 0.25) — sensemaking 강점.
- 외부 다중홉에서 그래프만 0.61~0.67로 **평면과 동급**(그래프-평면 격차 0.42→0.02 붕괴) → 연결이 실제 필요한 질의에 강함.
- `graphrag_e2b_hybrid`는 그래프+직접벡터라 **핀포인트도 유지**(내부 0.806)하며 연결성을 더함.

**주의(§10.9)**: 조직 문서가 **자동생성 near-duplicate**면 연결이 밀도만 높고 의미 없어 이득이 준다. **사람이 쓴 진짜 문서**일수록 이 프로필의 값어치가 산다.

**실행** (그래프 3단계: 추출 → 요약 → 평가)
```bash
ragbench index --method graphrag_e2b --config config/profile_community.yaml --data <corpus>
python scripts/build_community_summaries.py config/profile_community.yaml storage/graphrag_e2b
ragbench eval --method graphrag_e2b_hybrid --config config/profile_community.yaml --eval-set <set> --judge
# global·종합형 비중이 크면 기법을 graphrag_e2b_adaptive 로.
```

## 선택 기준 (결정 표)

| 상황 | 프로필 |
| --- | --- |
| 질문이 특정 사실/2홉, 문서 연결 약함 | 범용(`hybrid`) |
| 조직 문서가 서로 참조·연결 + global/종합 질의 | 커뮤니티(`graphrag_e2b_hybrid`) |
| global·"전사/각 부서/어떻게 이어지나" 가 다수 | 커뮤니티(`graphrag_e2b_adaptive`) |
| 속도·비용 최우선, 그래프 부담 | 범용(`hybrid`) |

> 근거 수치: [../CLAUDE.md](../CLAUDE.md) §10.4~10.9. 실행 상세: [GUIDE_RUN.md](GUIDE_RUN.md).
