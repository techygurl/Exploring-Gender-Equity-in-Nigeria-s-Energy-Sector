# Exploring Gender Equity in Nigeria's Energy Sector

This project grew out of a personal experience.

While seeking mentorship and hands-on opportunities as an aspiring energy data analyst, I began attending energy-related conferences and networking events. In those rooms—filled with expertise and ambition—I noticed a recurring theme:

> Gender inequality and limited career advancement opportunities were still very much present in the energy industry.

That realization motivated me to take a data-driven approach to understand the issue better.

Using 60 survey responses from professionals across the energy field, this project aims to explore:

- Trends in gender representation  
- Barriers to leadership  
- The role of mentorship and workplace equity perceptions  
- Obstacles and enablers to professional growth  
- Actionable, policy-level recommendations to promote gender equity  

---

## Core Research Questions

- What key factors contribute to women securing leadership positions?
- Does mentorship play a significant role in helping women reach leadership roles?
- Which energy subsectors exhibit stronger gender equity?
- Are women more likely to report stalled career growth due to gender bias?
- How does access to mentorship differ by gender, and does a mentor's gender matter?
- Is there a link between experience and leadership likelihood?
- Do men and women perceive workplace gender equity differently?
- How does educational background influence leadership opportunities?
- What systemic or organizational barriers are most commonly cited?

---

## Methods

To analyze gender equity in the energy sector, I used data from a custom survey targeting 60 professionals across Oil & Gas, Power, Renewables, and related subsectors.

**Tools & Libraries Used**:
- `pandas` – for data cleaning and transformation  
- `seaborn` & `matplotlib` – for data visualization  
- `scikit-learn` – for logistic regression and correlation testing

---

## Key Findings

### Does mentorship help women reach leadership roles?
<img width="2969" height="1763" alt="Impact of Key Factors to Leadership Roles by Gender (1)" src="https://github.com/user-attachments/assets/321bbf95-e89a-461a-91a7-a257f7cea345" />


- Women with mentors were 2x more likely to have held leadership positions in the energy sector.
- This effect remained significant after controlling for age, experience, education, role type, and sector.
- For men, the top factors were:
  - Experience (2.49x)
  - Education level (1.53x)
  - Having a mentor (1.5x)
- Experience is more influential for men, but mentorship still matters.

---

### Which energy subsectors show stronger gender equity?

| Sector        | Leadership (Female) | Leadership (Male) | Gender Gap |
|---------------|---------------------|-------------------|-------------|
| Power         | 80%                 | 83%               | 0.03        |
| Oil & Gas     | 45%                 | 65%               | 0.20        |

- Power Sector: Minimal gender gap, indicating strong gender equity  
- Oil & Gas: Significant gap, showing underrepresentation of women in leadership  
- Other subsectors excluded due to limited data

---

### How does access to mentorship differ by gender and does the gender of a mentor influence outcomes?

<img width="1607" height="1154" alt="Mentorship Access by Gender" src="https://github.com/user-attachments/assets/fad0f3d6-85a0-4d3b-bbbd-7471e0757e44" />


<img width="2968" height="1763" alt="Leadership Rate by Mentor Gender and Respondent Gender" src="https://github.com/user-attachments/assets/6676558b-5670-4bd2-a018-fb4156d2a30f" />

- Men with male mentors showed the highest leadership outcomes.
- Female mentors had a more balanced effect, slightly improving leadership outcomes for women.
- Same-gender mentorship may support leadership development, but the gender of the mentor showed no statistically dominant effect.

---

### Do men and women perceive workplace equity differently?

<img width="2705" height="1463" alt="Perception of Gender Equity by Gender and Experience" src="https://github.com/user-attachments/assets/0a6c778b-1817-4d5c-b054-bfc50bf2a94c" />

- Men consistently rate workplaces as more supportive of gender equity than women.
- The biggest perception gap exists in the 1–3 years experience range.
- As women gain experience, their perception of workplace equity improves, possibly due to:
  - Better advocacy
  - Greater opportunity
  - Adaptation to workplace culture

---

## Systemic & Organizational Barriers

Insights from energy sector leaders (especially in engineering and Oil & Gas) revealed key barriers:

1. **Underrepresentation of women in technical roles**  
   - Most women enter support functions (HR, admin, sales) rather than technical fields like engineering or field ops.
   
2. **Offshore/field work challenges**  
   - Seen as too strenuous or high-pressure, not because of ability, but due to rigid timelines and contracts.
   - Companies are hesitant to assign women to high-responsibility field roles.

3. **Male-dominated education pipelines**  
   - Engineering programs in many African countries remain male-heavy.
   - This imbalance feeds directly into the industry’s gender gap in leadership.

---

## Conclusion

Gender equity in the energy sector is not just a moral issue — it's a strategic imperative.

### Summary:

- Mentorship is a powerful tool: Women with mentors are twice as likely to reach leadership roles.
- Sector disparities exist: Power shows strong equity; Oil & Gas does not.
- Perception gaps: Women see less equity early in their careers, but views improve with experience.
- Education and field access matter: Technical paths to leadership are often blocked early.

To close the gap, we need inclusive mentorship programs, technical training access for women, and policy-driven support at the entry level.

---

## Let's Connect

- Questions or ideas? Reach out via [LinkedIn](https://www.linkedin.com/in/nissidouglas/)


---

## Repository Structure

