# Stakeholder validation of the cultural constraints

Instrument, protocol, and analysis plan for validating LMIC-HospitalNav's
cultural constraints with clinical staff at a Bangladeshi tertiary public
hospital.

---

## 1. What this fixes

The paper's headline claim is cultural, but the constraints were authored by the
research team from published proxemics literature. Section VIII currently
concedes this:

> *"Cultural constraints are encoded as predefined rules... site-specific
> calibration with local clinical and religious stakeholders... would further
> strengthen external validity."*

Acknowledging a validity gap is not closing it. A reviewer from the
CARESSES / CSSR4Africa lineage will ask on whose authority "robot present in the
prayer room" counts as a violation rather than an assumption.

**What this converts:** *"grounded in published proxemics literature"* →
*"validated with N clinical staff at a Bangladeshi tertiary public hospital."*

It is the single highest-leverage addition available to the paper, and it costs
days rather than GPU-months. It does not require DMCH specifically — Rajshahi
Medical College Hospital is the same class of overcrowded public tertiary
facility and is local to RUET.

---

## 2. Ethics approval — do this before collecting anything

**This is human-subjects research.** A survey of hospital staff about workplace
norms is minimal-risk, but it is still human-subjects data, and most journals
now require an ethics statement naming the approving body.

Required before the first response is collected:

1. **Institutional approval** — RUET's ethical review committee, or the
   equivalent. If no committee exists, obtain written approval from hospital
   administration and say so explicitly in the paper.
2. **Hospital permission** — from nursing administration for the ward(s).
3. **Informed consent** — verbal is acceptable for an anonymous minimal-risk
   survey, but the consent script must be delivered to every respondent.
4. **No identifying information.** Collect role, years of experience, and ward
   type only. No names, no employee numbers.

> Data collected without approval may be unpublishable, and no amount of later
> paperwork fixes it retroactively. Get the approval first.

Consent script (read aloud or printed at the top of the form):

> *We are researchers at RUET studying how a small delivery robot should behave
> in a hospital ward. We would like your opinion on a set of rules we have
> written for the robot. There are no right or wrong answers, and we are not
> evaluating you or your ward. Participation is voluntary, you may stop at any
> time, and we will not record your name or any information that identifies you.
> The responses will be published only as group totals. This will take about ten
> minutes. May we continue?*

---

## 3. Respondents

| | Target |
|---|---|
| Number | 15–20 (minimum 10 for the claim to hold up) |
| Roles | Ward nurses (majority), ward/nursing administrators, ward attendants |
| Wards | At least one male ward, one female ward, and one general ward |
| Exclusions | Anyone who has seen the simulator or the paper |

Mix of roles matters more than raw count. A reviewer will notice if all 20
respondents are nurses from a single ward.

**Administer in Bengali.** These are rules about local practice; asking in
English will bias both who responds and how they answer. Translate the
instrument below, and have a second Bengali speaker back-translate the items to
check that no rule changed meaning in transit.

---

## 4. The questionnaire

Present each rule in plain language, without the mathematics and without saying
which behaviour the robot currently implements. Two scales per rule, because
*"is this what actually happens here"* and *"should a robot be required to
follow it"* are different questions and reviewers will want them separated.

### Part A — Respondent background

- A1. Role: ☐ Nurse ☐ Ward administrator ☐ Ward attendant ☐ Other: ______
- A2. Years working in a hospital ward: ☐ <1 ☐ 1–3 ☐ 4–10 ☐ >10
- A3. Ward type where you mostly work: ☐ Male ☐ Female ☐ General ☐ Other

### Part B — The rules

For each, two ratings:

- **Scale M (matches practice):** 1 = strongly disagree … 5 = strongly agree
  — *"This describes what actually happens in this ward."*
- **Scale E (should be enforced):** 1 = not important … 5 = essential
  — *"A delivery robot should be required to follow this."*

| # | Rule as presented to the respondent | M | E |
|---|---|---|---|
| B1 | The robot should not drive through the prayer room, even when that is the shortest way to its destination. | 1–5 | 1–5 |
| B2 | Driving through the prayer room is more objectionable during the five daily prayer times than at other hours. | 1–5 | 1–5 |
| B3 | The robot should not enter a female ward unless its delivery task requires it. | 1–5 | 1–5 |
| B4 | The robot should not enter a male ward unless its delivery task requires it. | 1–5 | 1–5 |
| B5 | When a task does require entering a gendered ward, the robot should complete it and leave rather than remain in the ward. | 1–5 | 1–5 |
| B6 | The robot should keep a larger distance from a patient's family attendants than from hospital staff. | 1–5 | 1–5 |
| B7 | The robot should never pass closer than about an arm's length (0.6 m) to a family attendant. | 1–5 | 1–5 |
| B8 | The robot should slow down or wait rather than push through a group of family members standing together. | 1–5 | 1–5 |

### Part C — Parameters

- **C1.** Are these the prayer times at this hospital? Fajr 05:00–06:30,
  Dhuhr 12:00–13:00, Asr 15:00–16:00, Maghrib 17:30–18:30, Isha 19:00–20:00.
  ☐ Yes ☐ No — corrections: ______
- **C2.** Roughly how many family attendants accompany each patient in this
  ward? ☐ fewer than 1 ☐ about 1 ☐ 1–2 ☐ more than 2
  *(the simulator assumes about 1.2)*
- **C3.** At the busiest time of day, roughly how many people are in a ward
  corridor at once? ☐ <10 ☐ 10–25 ☐ 25–50 ☐ >50
- **C4.** Is there any part of the ward a delivery robot should never enter that
  we have not listed? ______

### Part D — Open questions

These matter most. They are where you find norms you did not model, and a
reviewer will read the answers as evidence you engaged with the setting rather
than assumed it.

- **D1.** If a small delivery robot worked in this ward, what would it do that
  staff, patients, or families would object to?
- **D2.** Which of the rules above is wrong, or would not work here?
- **D3.** Is there anything a robot should do differently for female patients
  than for male patients?
- **D4.** Who in the ward should be able to tell the robot to stop or move?

### Part E — Acceptance

- **E1.** Would a robot that carried medicines and specimens be useful in this
  ward? (1 = not at all … 5 = very useful)
- **E2.** Would you be comfortable working alongside one? (1 … 5)

---

## 5. Analysis

The ratings are **ordinal**. Report medians and interquartile ranges, not means
and standard deviations — a reviewer who checks will flag averaged Likert data,
and it is an easy thing to get right.

For each item compute: median M, IQR M, median E, IQR E, and the percentage of
respondents rating ≥ 4 ("agreement").

Code the open responses with a simple thematic pass: two people code
independently, compare, and report how many norms were raised that the
simulator does not model.

### The table this produces

Insert after the constraints subsection, referenced from the limitations
paragraph:

```latex
\begin{table}[t]
\caption{Stakeholder validation of the cultural constraints ($N=\,$?? clinical
staff, ?? hospital). ``Matches practice'' and ``Should be enforced'' are
median~[IQR] on a 1--5 ordinal scale; ``Agree'' is the share rating $\ge 4$.}
\label{tab:stakeholder}
\centering
\begin{tabular}{llcc c}
\toprule
& Constraint & Matches & Enforce & Agree \\
\midrule
B1 & Prayer-room avoidance          & ?\,[?--?] & ?\,[?--?] & ??\% \\
B2 & Prayer-time modulation         & ?\,[?--?] & ?\,[?--?] & ??\% \\
B3 & Female-ward entry              & ?\,[?--?] & ?\,[?--?] & ??\% \\
B4 & Male-ward entry                & ?\,[?--?] & ?\,[?--?] & ??\% \\
B5 & Transit, not dwelling          & ?\,[?--?] & ?\,[?--?] & ??\% \\
B6 & Attendant vs.\ staff spacing   & ?\,[?--?] & ?\,[?--?] & ??\% \\
B7 & 0.6\,m intrusion threshold     & ?\,[?--?] & ?\,[?--?] & ??\% \\
B8 & Yield to attendant groups      & ?\,[?--?] & ?\,[?--?] & ??\% \\
\bottomrule
\end{tabular}
\end{table}
```

### Honesty rules

**A low score is a finding, not a failure.** If respondents say the robot
passing through a female ward is unremarkable, then the gender constraint is
weaker than modelled, and the paper must say so — and ideally adjust the
weighting or report the discrepancy explicitly. A validation that confirms
everything reads as a validation that asked leading questions.

**Report the disagreements.** Wide IQRs are informative: they show the norm is
contested, which is itself a result about cultural constraints, and it is more
interesting than unanimity.

**Do not drop items that score badly.** Report all eight.

**N is small — claim accordingly.** With 15–20 respondents from one hospital,
the correct claim is *"consistent with the practice reported by clinical staff
at one Bangladeshi tertiary hospital,"* not *"validated for LMIC hospitals."*

---

## 6. What to change in the paper afterwards

1. Add `tab:stakeholder` after Section III-C.
2. Rewrite the *Generality of cultural constraints* paragraph
   ([main.tex:695](../manuscript/main.tex#L695)): replace the concession with the
   result, keeping the single-site limitation.
3. Add the ethics statement naming the approving body.
4. Add one sentence to the abstract only if the result is strong — the abstract
   is at 198 words against a ~200 limit, so it would need a trade.
5. **Add the stance sentence regardless of the outcome:** that gender
   segregation is modelled as observed institutional practice, not endorsed as
   a norm the authors prescribe. Without it, a reviewer can read the work as
   reifying the practice, and that is an avoidable rejection.

---

## 7. Timeline

| Step | Effort |
|---|---|
| Ethics + hospital permission | 1–2 weeks (the long pole — start now) |
| Translate and back-translate | 1 day |
| Collect 15–20 responses | 1–2 days on site |
| Analyse and code open responses | 1 day |
| Write the table and paragraph | 1 day |

The approval is the only slow part, and it is entirely front-loaded. Start it
before anything else.
