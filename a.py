# Sample email address
email = "abs@dfg.com"

# Split the email address at the "@" symbol
parts = email.split("@")

# Check if the email address has the "@" symbol and contains at least two parts
if len(parts) == 2:
    company_name = parts[1].split(".")[0]
    print(company_name)
else:
    print("Invalid email address")
