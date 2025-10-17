import torch
import transformers



tokenizer = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Base")

text = 'The Revolution Will Be Televised" - Jerry\'s stand-up comedy career takes a hit when the socialist government censors his material. Meanwhile, Elaine\'s boss at the publishing company insists on only printing pro-socialist literature, causing conflict with Elaine\'s personal beliefs. George tries to start a black market bagel business, but runs into trouble when the government cracks down on capitalism. "The People\'s Soup Kitchen" - Jerry and Elaine volunteer at a socialist soup kitchen, but their efforts are hampered by the kitchen\'s incompetent leader. Kramer becomes a member of the socialist party and starts dating a member of the revolution\'s propaganda committee. George becomes obsessed with getting a job at the newly nationalized golf course. "Equality for All" - Jerry is accused of being a capitalist pig when he charges extra for autographs at a comedy club. Elaine\'s new socialist boyfriend encourages her to give away all of her possessions, causing tension in their relationship. George becomes envious of a socialist neighbor who has access to better housing and resources. "The Great Collective Farm Experiment" - Jerry and Elaine visit a collective farm and accidentally destroy a crop, leading to a showdown with the angry farmers. Kramer becomes a leader in the socialist youth movement and starts organizing events and rallies. George joins a socialist theater group, but his lack of talent causes problems. "Comrades in Arms" - Jerry is drafted into the socialist army and becomes roommates with a group of quirky soldiers. Elaine starts a forbidden romance with a capitalist. George tries to start a socialist dating service, but his lack of understanding of socialist values leads to problems.'

text_chunks = text.split('"')


for chunk in text_chunks:
    num_tokens = len(tokenizer.encode(chunk))
    print(f"num_tokens: {num_tokens}")
    print(f"chunk: {chunk}")