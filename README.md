# ClinicalBertSnomed
This is some example code for using ClinicalBert to suggest SNOMED codes for medical documents.  
  
ClinicalBert is a large language model trained on clinical data and available as open source from Hugging Face.  

ClinicalBert is installed using Anaconda and an environment setup with Python 3.8 installed. This is pre-requisite for running this code.  
  
Python environment must be set to use the one created in Anaconda  
  
Important Note: GPU with CUDA cores will run way better than running this on the CPU, but it will run on the CPU. Check CUDA version in NVidia Control Panel.  
  
Install Anaconda and run Anaconda Terminal (MacOS)/Powershell (Windows) as Admin  

https://www.anaconda.com/products/distribution.
  
```conda create --name clinicalbert-env python=3.8 ``` (Important - anything above Python 3.8 not supported by pytorch below)  
  
```conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia ```
  
```pip3 install transformers ``` 
  
Can be run directly from VS Code by setting the python environment to clinicalbert-env from Anaconda  
  
Press CTRL+P in VS Code, type Python: Select Interpreter and select clinicalbert-env  
  
# Example response for discharge summary in the code

Entity found by AI: pyelonephritis, Label: problem, Score: 0.5424991250038147  
Matched SNOMED Code: 45816000  
Matched Term: Pyelonephritis  
  
Entity found by AI: hypertension, Label: problem, Score: 0.99832683801651  
Matched SNOMED Code: 38341003  
Matched Term: Hypertension  
  
Entity found by AI: type 2 diabetes, Label: problem, Score: 0.9944760799407959  
Matched SNOMED Code: 44054006  
Matched Term: Type 2 diabetes mellitus  
  
Entity found by AI: known complications, Label: problem, Score: 0.9750624895095825  
No results found.  
  
Entity found by AI: acute kidney injury, Label: problem, Score: 0.9971821904182434  
Matched SNOMED Code: 140031000119103  
Matched Term: Acute nontraumatic kidney injury  
  
Entity found by AI: transaminitis conditions, Label: problem, Score: 0.9037625789642334  
No results found.  
Entity found by AI: iron deficiency anemia, Label: problem, Score: 0.9983749389648438  
Matched SNOMED Code: 87522002  
Matched Term: Iron deficiency anaemia  
  
Entity found by AI: increased urinary frequency, Label: problem, Score: 0.9985155463218689  
Matched SNOMED Code: 162116003  
Matched Term: Frequent urination  
  
Entity found by AI: urgency, Label: problem, Score: 0.9971749782562256  
Matched SNOMED Code: 103391001  
Matched Term: Urgent  
  
Entity found by AI: dysuria, Label: problem, Score: 0.9937722086906433  
Matched SNOMED Code: 49650001  
Matched Term: Dysuria  
  
Entity found by AI: fever, Label: problem, Score: 0.9956320524215698  
Matched SNOMED Code: 386661006  
Matched Term: Fever  
  
Entity found by AI: rigors, Label: problem, Score: 0.9382731914520264  
Matched SNOMED Code: 38880002  
Matched Term: Rigor  
  
Entity found by AI: hypotensive, Label: problem, Score: 0.9978856444358826  
Matched SNOMED Code: 67763001  
Matched Term: Hypotensive episode  
  
Entity found by AI: tachycardic, Label: problem, Score: 0.9959766268730164  
Matched SNOMED Code: 3424008  
Matched Term: Tachycardia  
  
Entity found by AI: fever, Label: problem, Score: 0.9981398582458496  
Matched SNOMED Code: 386661006  
Matched Term: Fever  
  
Entity found by AI: urinary symptoms, Label: problem, Score: 0.8968126177787781  
Matched SNOMED Code: 249274008  
Matched Term: Urinary symptoms  
  
Entity found by AI: pyelonephritis, Label: problem, Score: 0.9972952008247375  
Matched SNOMED Code: 45816000  
Matched Term: Pyelonephritis  
  
Entity found by AI: fever, Label: problem, Score: 0.9946008920669556  
Matched SNOMED Code: 386661006  
Matched Term: Fever  
  
Entity found by AI: hypotensive, Label: problem, Score: 0.9976686835289001  
Matched SNOMED Code: 67763001  
Matched Term: Hypotensive episode  
  
Entity found by AI: initial assessment, Label: test, Score: 0.9659382104873657  
Matched SNOMED Code: 170888003  
Matched Term: ENT: initial assessment  
  
Entity found by AI: a blood pressure, Label: test, Score: 0.993086040019989  
Matched SNOMED Code: 466658000  
Matched Term: Blood pressure alarm  
  
Entity found by AI: serum lactate, Label: test, Score: 0.9927139282226562  
Matched SNOMED Code: 270982000  
Matched Term: Serum lactate measurement  
  
Entity found by AI: elevated, Label: problem, Score: 0.6571506857872009  
Matched SNOMED Code: 123126002  
Matched Term: Elevated  
  
Entity found by AI: a bolus of, Label: treatment, Score: 0.7119264602661133  
Matched SNOMED Code: 431393006  
Matched Term: Administration of intravenous fluid bolus  
  
Entity found by AI: iv fluid, Label: problem, Score: 0.8998098373413086  
Matched SNOMED Code: 118431008  
Matched Term: Intravenous fluid  
  
Entity found by AI: hypotensive, Label: problem, Score: 0.9834889769554138  
Matched SNOMED Code: 67763001  
Matched Term: Hypotensive episode  
  
Entity found by AI: an arterial line, Label: treatment, Score: 0.9133713245391846  
Matched SNOMED Code: 261241001  
Matched Term: Arterial line  
  
Entity found by AI: hemodynamic monitoring, Label: test, Score: 0.9935338497161865  
Matched SNOMED Code: 716777001  
Matched Term: Haemodynamic monitoring  
  
Entity found by AI: levophed, Label: test, Score: 0.6805927157402039  
Matched SNOMED Code: 35950011000036103  
Matched Term: Levophed  
  
Entity found by AI: crystalloids, Label: test, Score: 0.8787049055099487  
Matched SNOMED Code: 42076001  
Matched Term: Crystallin  
  
Entity found by AI: piptazo, Label: treatment, Score: 0.9253994226455688  
Matched SNOMED Code: 1177101000168109  
Matched Term: Piptaz 4 g/0.5 g (AFT)  
  
Entity found by AI: blood and urine cultures, Label: test, Score: 0.9898033142089844  
Matched SNOMED Code: 401324008  
Matched Term: Urine microscopy, culture and sensitivities  
  
Entity found by AI: serum lactate, Label: test, Score: 0.9843009114265442  
Matched SNOMED Code: 270982000  
Matched Term: Serum lactate measurement  
  
Entity found by AI: blood cultures, Label: test, Score: 0.9961587190628052  
Matched SNOMED Code: 30088009  
Matched Term: Blood culture  
  
Entity found by AI: e. coli, Label: problem, Score: 0.9949195981025696  
Matched SNOMED Code: 122002009  
Matched Term: E. coli detection  
  
Entity found by AI: all antibiotics, Label: treatment, Score: 0.9321900010108948  
Matched SNOMED Code: 109991000119100  
Matched Term: Antibiotic allergy  
  
Entity found by AI: oral ciprofloxacin, Label: treatment, Score: 0.9901398420333862  
Matched SNOMED Code: 324600002  
Matched Term: Ciprofloxacin 100 mg oral tablet  
  
Entity found by AI: antibiotics, Label: treatment, Score: 0.9957183003425598  
Matched SNOMED Code: 326779007  
Matched Term: Cytotoxic antibiotics  
  
Entity found by AI: symptoms of, Label: problem, Score: 0.841711699962616  
Matched SNOMED Code: 735641000  
Matched Term: Feigning of symptoms  
  
Entity found by AI: prostatism, Label: problem, Score: 0.6274069547653198  
Matched SNOMED Code: 11441004  
Matched Term: Prostatism  
  
Entity found by AI: an abdominal ultrasound, Label: test, Score: 0.9976039528846741  
Matched SNOMED Code: 432853001  
Matched Term: Ultrasound of anterior abdominal wall  
  
Entity found by AI: elevated liver enzymes, Label: problem, Score: 0.9987389445304871  
Matched SNOMED Code: 707724006  
Matched Term: Elevated liver enzymes level
