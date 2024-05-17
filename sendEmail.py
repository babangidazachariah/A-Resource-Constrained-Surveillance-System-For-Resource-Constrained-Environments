
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication

class SendEmail:
    def __init__(self, to_mail="babangidazachariah@gmail.com"):
        self.from_email = "applicationsurveillance@gmail.com"
        self.to_email = to_mail

        self.smtp_server = 'smtp.gmail.com'
        self.smtp_port = 587
        self.smtp_username = 'applicationsurveillance@gmail.com'
        self.smtp_password = 'tapcrouikdgrxotb'
        
    def SendMessage(self, subject, body, attach_path=None):
        if attach_path is None:
            message = f'Subject: {subject}\n\n{body}'

            with smtplib.SMTP(self.smtp_server, self.smtp_port) as smtp:
                smtp.starttls()
                smtp.login(self.smtp_username, self.smtp_password)
                smtp.sendmail(self.from_email, self.to_email, message)
        else:
            msg = MIMEMultipart()
            msg['From'] = self.from_email
            msg['To'] = self.to_email
            msg['Subject'] = subject
            msg.attach(MIMEText(body))

			
            #with open('report.pdf', 'rb') as f:
            with open(attach_path, 'rb') as f:
                ext = attach_path[attach_path.rfind('.'):]
                attachment = MIMEApplication(f.read(), _subtype=ext)
                attachment.add_header('Content-Disposition', 'attachment', filename=attach_path)
                msg.attach(attachment)

            with smtplib.SMTP(self.smtp_server, self.smtp_port) as smtp:
                smtp.starttls()
                smtp.login(self.smtp_username, self.smtp_password)
                smtp.send_message(msg)



#sndMail = SendEmail()
#sndMail.SendMessage('Class-based Send','This message was sent using the object-oriented implementation of the SendMail Class. This had an attachement.','copy_of_crate.obj')




