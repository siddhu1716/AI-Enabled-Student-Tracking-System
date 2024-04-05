from twilio.rest import Client

account_sid='AC8836dfeb51a6f5ea7f0b97cf4e7b2696'
auth_token = 'ade6965347c16992392b4af33c216e2d'

client = Client(account_sid, auth_token)
message_body = 'Hello, this is a test WhatsApp message!'

message = client.messages.create(
  from_='whatsapp:+14155238886',
  body=message_body,
  to='whatsapp:+919502152068',
)

print(f'WhatsApp message sent with SID: {message.sid}')
