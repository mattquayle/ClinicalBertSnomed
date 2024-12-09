# This is some example code for using ClinicalBert to suggest SNOMED codes for medical documents.
# ClinicalBert is a large language model trained on clinical data and available as open source from Hugging Face.
# ClinicalBert is installed using Anaconda and an environment setup with Python 3.8 installed. This is pre-requisite for running this code.
# Python environment must be set to use the one created in Anaconda
# Note GPU with CUDA cores will run way better than running this on the CPU, but it will run on the CPU. Check CUDA version in NVidia Control Panel.
# Install Anaconda and run Anaconda Terminal (MacOS)/Powershell (Windows) as Admin
# conda create --name clinicalbert-env python=3.8 (Important - anything above Python 3.8 not supported by pytorch below)
# conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
# pip3 install transformers
# Can be run directly from VS Code by setting the python environment to clinicalbert-env from Anaconda
from transformers import BertTokenizer, BertForTokenClassification
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForSeq2SeqLM
from transformers import pipeline
import requests

# Check if a GPU is available
import torch
device = 0 if torch.cuda.is_available() else -1  # -1 for CPU

# Define the base URL and endpoint for the SNOMED CT API
base_url = "https://r4.ontoserver.csiro.au/fhir/ValueSet/$expand"

# Load ClinicalBERT model and tokenizer. Model could run on a large instance in AWS with a lambda for the pipeline and nlp_ner
# Cost could be calculated based on number of documents filed per user per month.
model_name = "samrawal/bert-base-uncased_clinical-ner"  # A commonly used ClinicalBERT model on Hugging Face

#Bunch of alterative models that I looked at. Many don't work out the box and mt5 is a A LOT more complicated to use.
#model_name = "emilyalsentzer/Bio_ClinicalBERT" # Excellent for tasks like entity extraction, classification, or question answering on clinical data.
#model_name = "emilyalsentzer/Bio_Discharge_Summary_BERT"  # A variant of ClinicalBERT specifically fine-tuned on discharge summaries.
#model_name = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext" #Suitable for tasks where you need to link clinical terms to medical ontologies (e.g., SNOMED CT, UMLS).
#model_name = "dmis-lab/biobert-v1.1"
#model_name = "bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12"
#model_name = "kexinhuang123/medbert"
#model_name = "google/mt5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

# Initialize the pipeline with the device argument to use the GPU
# aggregation_strategy="first" tells the pipeline to match on the first part of the token it finds
# e.g. If a word like "diabetes" is split into sub-tokens ["di", "##abetes"], the model generates predictions for both di and ##abetes.
# The first strategy assigns the entity label of the first sub-token (di) to the entire word. Alternatives are max, average, simple. Simple splits tokens.
nlp_ner = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="first", device=device) 

# Test the model with sample discharge summary (example I found on the internet). We would need to extract the text from the document first.
text = """Patient Name: Smith, John 
NHS Number: 1234567 
VISIT ENCOUNTER            
Visit Number:  
DOB: 25-Dec-1950, 65 years old 
Gender: Male
 11186424686 
Admission Date:  
Discharge Date:  
08-Oct-2015       
14-Oct-2015 
Discharge Diagnosis: Pyelonephritis 
Primary Care Provider / Family Physician: 
Most Responsible Health Care Provider: 
Discharge Summary Completed by:  
Ensure Primary Care / 
Referring Provider 
information is correct 
Jay, Samantha; 416-555-5555 
Snow, Michael; Physician; 416-123-4567 
Lee, Dan; Senior Resident; 416-321-4567 on 23-Jul-2015 
Patient Encounter Type: Inpatient 
Discharge Disposition:  Discharged home from Toronto General Hospital (General Internal Medicine)
 DIAGNOSIS (Co-Morbidities and Risks)  
Conditions Impacting Hospital LOS: 
Pre-Existing: 
• Hypertension, Type 2 diabetes with no known complications 
Developed: 
• Acute kidney injury, Transaminitis  
Conditions Not Impacting LOS: 
• Iron deficiency anemia 
Risks: None
1. Upon arrival: Patient presented with five days of increased urinary frequency, urgency and dysuria as well as 
48 hours of fever and rigors. He was hypotensive and tachycardic upon arrival to the emergency department. 
The internal medicine service was consulted. The following issues were addressed during the hospitalization: 
Summary Course in Hospital (Issues Addressed): 
2. Fever and urinary symptoms: A preliminary diagnosis of pyelonephritis was established. Other causes of fever 
were possible but less likely. The patient was hypotensive on initial assessment with a blood pressure of 
80/40. Serum lactate was elevated at 6.1. A bolus of IV fluid was administered (1.5L) but the patient remained 
hypotensive. Our colleagues from ICU were consulted. An arterial line was inserted for hemodynamic 
monitoring. Hemodynamics were supported with levophed and crystalloids. Piptazo was started after blood 
and urine cultures were drawn. After 12 hours serum lactate had normalized and hemodynamics had 
stabilized. Blood cultures were positive for E.Coli that was sensitive to all antibiotics. The patient was stepped 
down to oral ciprofloxacin to complete a total 14 day course of antibiotics. 
On further review it was learned that the patient has been experiencing symptoms of prostatism for the last 
year. An abdominal ultrasound performed for elevated liver enzymes and acute kidney injury confirmed a 
Printed by: Snow, Mike on 15-OCT-2015 
SAMPLE 
Page 2 of 3 
severely enlarged prostate. Urinary retention secondary to BPH was the likely underlying mechanism that 
contributed to the development of pyelonephritis in this patient. He was started on Tamsulosin 0.4mg PO qhs 
and tolerated it well with no orthostatic intolerance. Post void residuals show 150-200cc of retained urine in 
the bladder. An outpatient referral to Urology has been requested by our team. 
3. Elevated liver enzymes and creatinine. Both of these were thought to be related to end organ hypoperfusion 
in the setting of sepsis. Values improved with the administration of IV fluid and stabilization of the patients 
hemodynamics. Abdominal ultrasound with doppler flow and urine analysis ruled out other possible 
etiologies. Liver enzymes remain slightly above normal values at the time of discharge. We ask that the 
patients’ family physician repeat these tests in 2 weeks’ time to ensure complete resolution. 
Investigations:  
Labs 
Test 
Include important developments while in 
hospital (do not be over-inclusive)  
Test Date 
Results 
1 Lactate 
Units 
08-Oct-2015 
6.1 
2 ALP 
mmol/L 
08-Oct-2015 
450 
3 ALT 
IU/L 
08-Oct-2015 
1001 
4 AST 
IU/L 
08-Oct-2015 
850 
5 Bilirubin 
08-Oct-2015 
24 
6 INR 
08-Oct-2015 
1.1  
7 Creatinine 
08-Oct-2015 
170 
8 ALP 
14-Oct-2015 
35 
9 ALT 
14-Oct-2015 
90 
10 AST 
14-Oct-2015 
70 
11 Bilirubin 
14-Oct-2015 
17 
12 Creatinine 
14-Oct-2015 
Radiology:  
Test 
Test Date 
Interventions (Procedures & Treatments):  
1.  Arterial line insertion 
Allergies:  
IU/L 
umol/L 
umol/L 
IU/L 
IU/L 
IU/L 
umol/L 
66 
Results 
umol/L 
Only include significant or 
abnormal lab, radiology 
and diagnostic results 
08-Oct-2015 
1 Abdominal and Pelvic 
Ultrasound 
Impression: Normal kidneys, liver and 
doppler analysis. Enlarged prostate. 
• Latex – Causes rashes 
DISCHARGE PLAN  
Medications at Discharge: 
Unchanged Medications:  
 Proferrin 1 tablet po daily 
 Ramipril 10mg po daily 
 Metformin 500mg po BID 
Adjusted Medications: 
None 
Printed by: Snow, Mike on 15-OCT-2015 
Categorized listing of medications 
SAMPLE 
Page 3 of 3 
New Medications: 
 Ciprofloxacin 500 mg twice daily for 7 days 
 Tamsulosin 0.4mg po QHS  
Discontinued Medications: 
None  
Follow-Up Instructions for Patient: 
Do not exceed more 
than three pages! 
1. Fever and urinary symptoms: Should these symptoms return please contact your  
family doctor urgently or visit your nearest emergency department.  
2. Dizziness: You have been started on a new medication for your enlarged prostate. 
If you experience dizziness upon sitting or standing please contact your family  
physician. 
Follow-Up Plan Recommended for Receiving Providers: 
1. Dear Dr. Jay:  Your patient was admitted to hospital with a diagnosis of pyelonephritis complicated by 
acute kidney injury and transaminitis. He likely has BPH which contributed to this. We have asked him to 
arrange follow up with you in two weeks’ time. Please repeat his AST and ALT at that time to ensure that 
they have normalized. We have also referred him to our colleagues in urology for further assessment of 
his prostate. """

#Execute the NLP via the model
entities = nlp_ner(text)

#remove entities where confidence score is lower than 0.9 to remove inaccuracies
filtered_entities = [entity for entity in entities if float(entity['score']) >= 0.9]

#remove entities where length of the term is less than 3 - must have more than 3 characters for a good match
filtered_entities = [entity for entity in entities if len(entity["word"]) > 3]

#For each matched entity, search SNOMED for a matching term and return the first result. This could be a smarter algorithm.
#Here I'm using a free SNOMED CT API, but we would used MKB to search on term. Pretty good hits on the below.
for entity in filtered_entities:
    print(f"Entity found by AI: {entity['word']}, Label: {entity['entity_group']}, Score: {entity['score']}")
    # Define the parameters for the GET request
    params = {
        "url": "http://snomed.info/sct?fhir_vs",
        "filter": {entity['word']}
    }

    # Make the GET request
    response = requests.get(base_url, params=params)

    # Check the response
    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()
        
        # Retrieve the first response (if available)
        if "expansion" in data and "contains" in data["expansion"]:
            first_item = data["expansion"]["contains"][0]  # Get the first item
            print(f"Matched SNOMED Code: {first_item.get('code')}")
            print(f"Matched Term: {first_item.get('display')}")
            print(f"")
        else:
            print("No results found.")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

