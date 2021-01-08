import smtplib
from email.message import EmailMessage
import io
import pandas

def send_email(subject, body, df=False, html=False,
               _from='dev.jonathan.coignard@gmail.com',
               _pwd='idontknow'):
    """
    Convenient function to get automatic updates on simulations
    """
    message = EmailMessage()
    message['From'] = _from
    message['To'] = _from
    message['Subject'] = subject
    message.set_content(body)

    if isinstance(df, pandas.DataFrame):
        message.add_attachment(_export_csv(df), filename='result.csv')

    if html:
        with open(html, 'r') as f:
            message.add_attachment(f.read(), filename=html)

    # Send
    mail_server = smtplib.SMTP_SSL('smtp.gmail.com')
    mail_server.login(_from, _pwd)
    mail_server.send_message(message)
    mail_server.quit()
    return True

def _export_csv(df):
  with io.StringIO() as buffer:
    df.to_csv(buffer)
    return buffer.getvalue()
