from twilio.rest import Client

account_sid='Your Account SID'
auth_token = 'Your Token'

client = Client(account_sid, auth_token)
message_body = 'Hello, this is a test WhatsApp message!'

message = client.messages.create(
  from_='whatsapp:+14155238886',
  body=message_body,
  to='whatsapp:+919502152068',
)

print(f'WhatsApp message sent with SID: {message.sid}')
