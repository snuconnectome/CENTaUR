# Data Use Agreement (DUA) Template
## Ko-CENTaUR Multi-Lab Data Sharing

**Agreement Date**: [Date]

**Between**:
- **Provider Institution**: [Collaborating Lab Name]
- **Recipient Institution**: [PI Institution]

---

## 1. Purpose

This Data Use Agreement ("Agreement") governs the transfer and use of de-identified behavioral and clinical assessment data from Provider Institution to Recipient Institution for the Ko-CENTaUR research project.

**Project Title**: Korean Cognitive Model Development through Multi-Lab Data Aggregation

**Principal Investigator**: [PI Name], [PI Institution]

**IRB Protocol Number**: [To be assigned]

---

## 2. Data Description

### 2.1 Data to be Shared

Provider Institution agrees to share the following de-identified data:

**Assessment Data**:
- [ ] K-MMSE (Korean Mini-Mental State Examination)
- [ ] PHQ-9 (Patient Health Questionnaire)
- [ ] GAD-7 (Generalized Anxiety Disorder)
- [ ] SDQ (Strengths and Difficulties Questionnaire)
- [ ] CBCL (Child Behavior Checklist) - summary scores only
- [ ] Other: ___________________

**Participant Metadata** (de-identified):
- Age (in months)
- Gender
- Clinical diagnosis (broad category)
- Assessment date (year only)
- Lab identifier (assigned by Recipient)

**Estimated Data Volume**:
- Number of participants: _______
- Age range: _______ to _______
- Clinical vs. Control ratio: _______
- Assessment types included: _______

### 2.2 Data Format

Data will be provided in one of the following formats:
- [ ] JSONL (preferred)
- [ ] CSV with data dictionary
- [ ] Excel with standardized column names
- [ ] Other: ___________________

---

## 3. De-identification Requirements

### 3.1 Provider Responsibilities

Provider Institution certifies that:

1. **Original IRB Approval**: All data was collected under appropriate IRB approval with participant informed consent

2. **De-identification**: All 18 HIPAA identifiers have been removed:
   - Names, addresses, phone numbers, email addresses
   - Dates (except year), Social Security numbers
   - Medical record numbers, account numbers
   - Geographic subdivisions smaller than state
   - Photos, biometric identifiers
   - Any other unique identifiers

3. **Quality Control**: Data has been checked for completeness and accuracy

4. **Legal Authority**: Provider has legal authority to share this data

### 3.2 Recipient Responsibilities

Recipient Institution agrees to:

1. **Verification**: Conduct secondary de-identification verification
2. **No Re-identification**: Not attempt to identify individual participants
3. **Reporting**: Immediately report any inadvertent discovery of identifiable information
4. **Correction**: Work with Provider to correct any identification issues

---

## 4. Permitted Uses

### 4.1 Authorized Activities

Recipient Institution may use the data for:

1. **Ko-CENTaUR Model Training**: Train cognitive models on behavioral data
2. **Validation**: Test model performance on Korean normative data
3. **Research Publications**: Publish aggregated results in peer-reviewed journals
4. **Model Release**: Release trained public adapter model to research community

### 4.2 Prohibited Activities

Recipient Institution shall NOT:

1. **Re-identification**: Attempt to identify individual participants
2. **Unauthorized Sharing**: Share raw data with third parties without written consent
3. **Commercial Use**: Use data for commercial purposes without separate agreement
4. **Combined Datasets**: Link with other datasets that could enable re-identification

---

## 5. Data Security and Storage

### 5.1 Security Measures

Recipient Institution agrees to implement:

1. **Encryption**: AES-256 encryption for data at rest
2. **Access Control**: Limit access to PI and approved research staff (maximum 3 individuals)
3. **Two-Factor Authentication**: Required for all data access
4. **Audit Logs**: Maintain logs of all data access
5. **Secure Transfer**: Use encrypted channels (SFTP, encrypted email) for data transfer

### 5.2 Data Location

Data will be stored at:
- **Primary Location**: [Institutional secure server]
- **Backup Location**: [Institutional backup system]
- **Access**: On-site only (no cloud storage without encryption)

### 5.3 Data Retention and Destruction

- **Retention Period**: 10 years post-publication
- **Destruction Method**: Secure deletion meeting DOD 5220.22-M standard
- **Destruction Certificate**: Provided to Provider upon request

---

## 6. Publication and Authorship

### 6.1 Co-Authorship

Provider Institution will receive:

1. **Co-authorship**: On primary Ko-CENTaUR publication
   - Author representation: [Primary contact name]
   - Affiliation: [Provider Institution]

2. **Acknowledgment**: In all subsequent papers using the model

3. **Model Access**: Free access to trained Ko-CENTaUR model for research use

### 6.2 Publication Review

- Recipient will share manuscripts with Provider **30 days** before submission
- Provider has **14 days** to review and provide comments
- Recipient will consider all comments in good faith
- Provider can request delay for patent/IP considerations (max 90 days)

### 6.3 Data Attribution

All publications will include:
- Acknowledgment of data contribution from Provider Institution
- Statement: "Data provided by [Provider Institution] under Data Use Agreement"
- Source lab anonymized in public datasets (LAB1, LAB2, etc.)

---

## 7. Intellectual Property

### 7.1 Data Ownership

- Provider retains ownership of original data
- Recipient owns trained model weights and code
- Joint ownership of derivative datasets (processed/standardized data)

### 7.2 Model Licensing

Trained Ko-CENTaUR model will be released under:
- **License**: Apache 2.0 (or more permissive)
- **Public Adapter**: Freely available on HuggingFace Hub
- **Private Adapter**: Restricted to collaborating institutions

### 7.3 Patents and IP

- No patents will be filed that restrict research use of the model
- Commercial licensing agreements will include revenue sharing if data significantly contributed (>20% of training data)

---

## 8. Breach and Compliance

### 8.1 Security Breach Protocol

In the event of data breach or unauthorized access, Recipient will:

1. **Immediate Notification**: Notify Provider within 24 hours
2. **Investigation**: Conduct thorough investigation of breach
3. **Mitigation**: Implement immediate remediation measures
4. **Reporting**: Provide written report within 5 business days
5. **Regulatory**: Comply with all regulatory reporting requirements

### 8.2 Non-Compliance Consequences

Failure to comply with this Agreement may result in:
- Immediate termination of data use rights
- Return or destruction of all data
- Exclusion from authorship
- Legal action for damages

---

## 9. Term and Termination

### 9.1 Term

This Agreement becomes effective upon signing and remains in effect for:
- **Primary Term**: Duration of Ko-CENTaUR project (12 months)
- **Extended Term**: 10 years for data retention and follow-up studies

### 9.2 Termination

Either party may terminate this Agreement with:
- **Notice Period**: 30 days written notice
- **Data Return**: Recipient will return or destroy all data within 30 days of termination
- **Publication Rights**: Preserved for work completed before termination

---

## 10. Miscellaneous

### 10.1 Amendments

This Agreement may be amended only by written agreement signed by both parties.

### 10.2 Governing Law

This Agreement shall be governed by the laws of [Jurisdiction].

### 10.3 Dispute Resolution

Disputes will be resolved through:
1. Good faith negotiation (30 days)
2. Mediation (if negotiation fails)
3. Arbitration or litigation (last resort)

### 10.4 Entire Agreement

This Agreement constitutes the entire agreement between parties regarding the subject matter.

---

## 11. Signatures

### Provider Institution

**Authorized Signatory**:

Name: ___________________________
Title: ___________________________
Institution: ______________________
Signature: ________________________
Date: ____________________________

**Principal Investigator** (if different):

Name: ___________________________
Email: ___________________________
Phone: ___________________________

### Recipient Institution

**Authorized Signatory**:

Name: ___________________________
Title: ___________________________
Institution: ______________________
Signature: ________________________
Date: ____________________________

**Principal Investigator**:

Name: ___________________________
Email: ___________________________
Phone: ___________________________

---

## Appendices

**Appendix A**: Data Dictionary and Variable Definitions
**Appendix B**: Data Transfer Protocol
**Appendix C**: De-identification Checklist
**Appendix D**: Security Incident Response Plan

---

## Contact Information

**For questions about this Agreement**:
- Recipient PI: [Email, Phone]
- Provider PI: [Email, Phone]
- Institutional IRB: [Email, Phone]

**Effective Date**: [Date of last signature]

**Review Date**: [12 months from effective date]
