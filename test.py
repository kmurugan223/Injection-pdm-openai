from openai import OpenAI
 
client = OpenAI()
 
response = client.responses.create(

  model="gpt-3.5-turbo",

  input="Tell me a three sentence bedtime story about a unicorn."

)
 
print(response.output[0].content[0].text)

 