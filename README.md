📩 Automated AI Email Response System

An AI-powered email assistant that automatically processes incoming emails, analyzes content, and generates relevant responses using NLP (BERT, TF-IDF).

🚀 Features

✅ Automated Email Processing – Reads and categorizes incoming emails
✅ AI-Powered Responses – Uses BERT & TF-IDF for context-based replies
✅ Keyword Filtering – Identifies relevant emails using predefined keywords
✅ Secure Authentication – Uses environment variables for email credentials
✅ Scheduled Inbox Checking – Runs periodically to handle incoming emails

🛠 Tech Stack
	•	Python – Core programming language
	•	IMAPClient & smtplib – Email retrieval and sending
	•	BERT (Transformers) – AI model for intelligent email responses
	•	TF-IDF & Cosine Similarity – Matching incoming emails to past queries
	•	Pandas – Data handling for email logs

🔧 Setup Instructions

1️⃣ Clone the Repository

git clone https://github.com/arjunghosh4/Email-Project.git

cd email-response-system

2️⃣ Install Dependencies

pip install -r requirements.txt

3️⃣ Configure Environment Variables

Create a .env file in the root directory and add:

IMAP_SERVER=imap.example.com
IMAP_PORT=993
EMAIL_USER=your_email@example.com
EMAIL_PASS=your_password
SMTP_SERVER=smtp.example.com
SMTP_PORT=465

4️⃣ Run the System

python email_response.py

📊 How It Works
	1.	Connects to the email inbox via IMAP
	2.	Filters emails based on keywords
	3.	Extracts email subject & body
	4.	Uses TF-IDF & BERT to find the best response
	5.	Sends a reply via SMTP

📌 Example Usage

🔹 Incoming Email:

	Subject: Issue
Body: I’m facing invoice issue.

🔹 AI Response:

	“Predicted Category: Billing and Payments | AI Response: I am currently investigating the discrepancy in your digital services invoice. To better assist you, could you kindly provide your account number and the specific invoice number? I will review the details and will work on issuing a corrected invoice promptly. If necessary, I may need to contact you at a convenient time for further discussion. Please let me know a suitable time to call you. Your patience and cooperation on this matter are greatly appreciated.”

🤖 Future Improvements
	•	Support for multi-language email responses
	•	Advanced sentiment analysis for personalized replies
	•	Integration with CRM tools like Salesforce

🤝 Contributions

Feel free to submit issues and pull requests. Let’s build something awesome together! 🚀
