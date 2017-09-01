import requests_oauthlib as OA

client_id = r'dj0yJmk9azFZU3E2eDQwejYwJmQ9WVdrOWFUbHpRM0pKTTJVbWNHbzlNQS0tJnM9Y29uc3VtZXJzZWNyZXQmeD01NA--'
client_secret = r'c3da9fef03ab09a049d322a9aa31f9c2c1a5ffc4'
redirect_uri = 'oob'

scope = ['fspt-w']
oauth = OA.OAuth2Session(client_id, redirect_uri=redirect_uri,scope=scope)
authorization_url, state = oauth.authorization_url(
    'https://api.login.yahoo.com/oauth2/request_auth',
    response_type=["id_token"], nonce="YihsFwGKgt3KJUh6tPs2")

print('Please go to %s and authorize access.' % authorization_url)
authorization_response = raw_input('Enter the full callback URL')