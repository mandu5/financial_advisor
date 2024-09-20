# financial_advisor
Financial Advisor for Personalized Investment Strategies


# Model Summary

The Financial Advisor Chatbot Model is a specialized language model fine-tuned from Google's gemma-2-2b-it. By analyzing user-provided financial data, the chatbot generates tailored investment strategies, budgeting recommendations, and financial planning insights to assist users in making informed financial decisions.

If you want more information, please visit https://www.kaggle.com/code/mandu5/financial-advisor/notebook

## Usage
```python
# Install necessary libraries
!pip install transformers --quiet

# Import libraries
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the model and tokenizer
model_name = '/kaggle/input/financial-advisor/transformers/gemma-2-2bfinancial-advisor-chating-version-0.1/1/financial_advisor_pretrained'

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
)

# Move the model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define the function to generate advice (same as before)
def get_financial_advice(user_profile, model=model, tokenizer=tokenizer):
    prompt = f"input: {user_profile}\noutput:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=150,
        no_repeat_ngram_size=2,
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split('output:')[-1].strip()

# Example usage
user_profile = """
User Profile:
- Gender: Female
- Age: 34
- Investment Avenues Interested: Yes
- Preferred Investments:
  - Mutual Funds (Preference: 1)
  - Equity Market (Preference: 2)
  - Debentures (Preference: 5)
  - Government Bonds (Preference: 3)
  - Fixed Deposits (Preference: 7)
  - PPF (Preference: 6)
  - Gold (Preference: 4)
- Investment Objectives: Capital Appreciation
- Investment Purpose: Wealth Creation
- Investment Duration: 1-3 years
- Expected Returns: 20%-30%
- Savings Objective: Retirement Plan
- Source of Information: Newspapers and Magazines

Question:
What investment strategies should I consider?
"""

print("User Profile and Question:")
print(user_profile)
print("\nGenerated Advice:")
print(get_financial_advice(user_profile))
```


## Output
#### User Profile and Question:
- Gender: Male
- Age: 27
- Investment Avenues Interested: Yes
- Preferred Investments:
  - Mutual Funds (Preference: 4)
  - Equity Market (Preference: 5)
  - Debentures (Preference: 1)
  - Government Bonds (Preference: 2)
  - Fixed Deposits (Preference: 7)
  - PPF (Preference: 3)
  - Gold (Preference: 6)
- Investment Objectives: Growth
- Investment Purpose: Wealth Creation
- Investment Duration: 1-3 years
- Expected Returns: 10%-20%
- Savings Objective: Education
- Source of Information: Television

#### Question:
What investment strategies should I consider?

## Generated Advice:
Considering your objectives of Growth and Wealth creation over 
1 to 	3  years, you might explore investment avenues like Mutual Fund. Given your expected returns of 

15%-30%, these options align with your goals. Remember to diversify your portfolio and assess the risks involved. Consulting a financial advisor can provide personalized guidance.

What investments should you consider?:
Mutual Fund

Please note that this output is generic and does not provide specific investment recommendations. It suggests exploring investment options like mutual funds based on your growth objectives. You should consider your individual circumstances and goals before making investment decisions.
