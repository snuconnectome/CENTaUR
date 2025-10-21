# IRB Protocol: Ko-CENTaUR Multi-Lab Data Sharing Study

**Protocol Title**: Korean Cognitive Model Development through Multi-Lab Data Aggregation
**Protocol Type**: Expedited Review (Secondary Data Analysis)
**Study Duration**: 12 months
**Principal Investigator**: [PI Name, Institution]

---

## 1. Study Overview

### 1.1 Background and Significance

Recent advances in large language models (LLMs) have enabled the development of cognitive models that can predict human behavior in psychological experiments. The CENTaUR framework (Binz & Schulz, 2023, PNAS) demonstrated that LLMs fine-tuned on human experimental data can effectively model cognitive processes.

However, existing cognitive models are primarily English-language based and trained on Western populations, limiting their applicability to Korean clinical and developmental populations. This study aims to develop Ko-CENTaUR, a Korean-language cognitive model trained on aggregated de-identified data from multiple Korean research laboratories.

### 1.2 Study Objectives

**Primary Objective**: Develop and validate a Korean-language cognitive model (Ko-CENTaUR) that can predict cognitive and clinical measures from behavioral data

**Secondary Objectives**:
1. Establish Korean normative data for cognitive assessments across developmental stages
2. Validate model performance in clinical vs. healthy control discrimination
3. Create a shareable public adapter for research use

### 1.3 Study Design

**Design**: Retrospective secondary data analysis with multi-lab data aggregation

**Data Sources**: De-identified existing datasets from 3-5 collaborating Korean research laboratories

**No New Data Collection**: This study involves ONLY secondary analysis of existing de-identified data. No new participants will be recruited or assessed.

---

## 2. Participant Information

### 2.1 Data Sources

**Source**: Existing de-identified datasets from collaborating research laboratories

**Estimated Total N**: 3,000-5,000 sessions across all labs

**Age Range**: 0-18+ years (developmental span)
- Children (6-11 years): N ≈ 800-1,200
- Adolescents (12-17 years): N ≈ 800-1,200
- Young Adults (18-25 years): N ≈ 800-1,200
- Adults (26+ years): N ≈ 600-1,200

**Clinical Distribution**:
- Healthy controls: 60-70%
- Clinical samples: 30-40% (ADHD, anxiety, depression, ASD, etc.)

### 2.2 Inclusion Criteria (for Original Data)

- Korean language speakers
- Completed at least one standardized assessment (K-MMSE, PHQ-9, SDQ, CBCL, etc.)
- Data collected with appropriate IRB approval at source institution
- Data can be fully de-identified per HIPAA standards

### 2.3 Exclusion Criteria

- Data containing identifiable information that cannot be removed
- Data collected without proper informed consent at source institution
- Incomplete or corrupted data files

---

## 3. Data Sharing and De-identification

### 3.1 Data Use Agreements (DUAs)

Each collaborating laboratory will:
1. Sign a Data Use Agreement (DUA) with PI's institution
2. Verify that original data collection had appropriate IRB approval
3. Provide only fully de-identified datasets
4. Receive co-authorship on resulting publications

### 3.2 De-identification Procedures

**HIPAA Identifiers Removed**:
All 18 HIPAA identifiers will be removed before data transfer:
1. Names
2. Geographic subdivisions smaller than state
3. Dates (except year)
4. Phone numbers
5. Email addresses
6. Social Security numbers
7. Medical record numbers
8. Health plan beneficiary numbers
9. Account numbers
10. Certificate/license numbers
11. Vehicle identifiers
12. Device identifiers and serial numbers
13. Web URLs
14. IP addresses
15. Biometric identifiers
16. Full-face photos
17. Other unique identifying numbers/codes
18. Any other unique identifying number, characteristic, or code

**Verification Process**:
1. Source lab performs initial de-identification
2. PI team conducts secondary verification check
3. Automated scripts scan for remaining identifiers
4. Manual review of 10% random sample

### 3.3 Data to be Shared

**Behavioral Data**:
- Assessment responses (K-MMSE, PHQ-9, SDQ, CBCL)
- Task performance data
- Reaction times and choice patterns

**Metadata** (de-identified):
- Age (in months)
- Gender
- Clinical diagnosis (broad category only, e.g., "ADHD" not specific subtype)
- Assessment scores (T-scores, percentiles)

**NOT Shared**:
- Names, addresses, contact information
- Specific dates (only year retained)
- Detailed clinical notes
- School or clinic names
- Any identifiable information

---

## 4. Data Management and Security

### 4.1 Data Storage

**Location**: Institutional secure server with restricted access

**Encryption**: AES-256 encryption for data at rest

**Access Control**:
- PI + approved research staff only (2-3 individuals)
- Two-factor authentication required
- Access logs maintained

### 4.2 Data Retention

**Retention Period**: 10 years post-publication (per institutional policy)

**Destruction**: Secure deletion after retention period

### 4.3 Data Sharing Beyond Study

**Public Release**:
- Public adapter model will be released (HuggingFace Hub)
- NO raw participant data will be publicly released
- Only aggregated/summary statistics in publications

**Private Adapter**:
- Private adapter (trained on sensitive clinical data) will NEVER be released
- Restricted to PI institution for research use only

---

## 5. Risks and Benefits

### 5.1 Risks

**Minimal Risk Classification**: This study qualifies for minimal risk as:
1. Only de-identified existing data is used
2. No new data collection or participant contact
3. No identifiable information in study database
4. Standard data security measures in place

**Potential Risks**:
- Theoretical risk of re-identification (mitigated by thorough de-identification)
- Data breach risk (mitigated by encryption and access controls)

### 5.2 Benefits

**Direct Benefits**: None to individual participants (secondary data analysis)

**Societal Benefits**:
1. Improved cognitive assessment tools for Korean populations
2. Validated Korean normative data across developmental stages
3. Enhanced clinical screening capabilities
4. Open-source research tool for Korean cognitive science

---

## 6. Statistical Considerations

### 6.1 Sample Size Justification

**Target**: N = 3,000-5,000 sessions

**Rationale**:
- Based on original CENTaUR (Binz & Schulz, 2023): N = 60,092 sessions showed robust performance
- Multi-lab aggregation provides diversity in assessment protocols
- Sufficient for stratified analysis by age group and clinical status
- Adequate for train/validation/test splits (80%/10%/10%)

### 6.2 Statistical Analysis Plan

**Primary Analysis**: Model performance metrics
- Negative Log-Likelihood (NLL) on held-out test set
- Token-level accuracy
- Korean norm correlation (Pearson r)

**Secondary Analysis**:
- Age effect validation (developmental trajectories)
- Clinical discrimination (AUC for clinical vs. control)
- Cross-task generalization

---

## 7. Ethical Considerations

### 7.1 Multi-Lab Collaboration

**Incentive Structure**:
- Co-authorship for contributing labs
- Access to trained Ko-CENTaUR model for research use
- No financial compensation

**Fairness**:
- All contributing labs receive equal recognition
- Contribution acknowledged in publications
- Model access provided to all collaborators

### 7.2 Clinical Use Considerations

**Research Tool Only**:
- Ko-CENTaUR is intended for research purposes
- NOT a diagnostic tool
- Clinical use requires local validation and regulatory approval

**Bias Mitigation**:
- Diverse lab sources reduce single-site bias
- Age and gender stratification in sampling
- Clinical vs. control balance monitored

### 7.3 Data Provenance

**Lab Attribution**:
- Source lab tracked in metadata (LAB1, LAB2, etc.)
- Enables sensitivity analysis by data source
- Allows exclusion if data quality issues identified

---

## 8. Timeline

| Phase | Duration | Key Activities |
|-------|----------|----------------|
| IRB Approval | Weeks 1-6 | Submit protocol, respond to queries |
| DUA Negotiation | Weeks 3-8 | Finalize agreements with 3-5 labs |
| Data Transfer | Weeks 8-12 | Receive de-identified datasets |
| Data Validation | Weeks 12-16 | Quality checks, standardization |
| Model Training | Weeks 16-40 | Iterative training and validation |
| Analysis | Weeks 40-48 | Comprehensive evaluation |
| Dissemination | Weeks 48-52 | Publication and model release |

---

## 9. References

1. Binz, M., & Schulz, E. (2023). Using cognitive psychology to understand GPT-3. Proceedings of the National Academy of Sciences, 120(6), e2218523120.

2. DHHS. (2009). Guidance Regarding Methods for De-identification of Protected Health Information in Accordance with the Health Insurance Portability and Accountability Act (HIPAA) Privacy Rule.

---

## 10. Appendices

**Appendix A**: Data Use Agreement Template
**Appendix B**: De-identification Checklist
**Appendix C**: Data Security Plan
**Appendix D**: Collaborating Lab Contact List (to be completed)

---

## IRB Application Checklist

- [ ] Protocol narrative (this document)
- [ ] Data Use Agreement template
- [ ] De-identification procedures
- [ ] Data security plan
- [ ] Letters of collaboration from partner labs
- [ ] PI CV and training certificates
- [ ] Conflict of interest disclosure
- [ ] IRB application fee ($500)

**Expected Review Timeline**: 4-6 weeks (expedited review for secondary data analysis)

**Decision Criteria**: If IRB not submitted by Week 4 → Escalate to PI
