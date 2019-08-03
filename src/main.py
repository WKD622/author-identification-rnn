from library.files.files import FileLoader

embed_size = 150
hidden_size = 1024
num_layers = 1
num_epochs = 20
batch_size = 20
timesteps = 30
learning_rate = 0.002
number_of_authors = 3

base_path = '../data'
files = []

for i in range(number_of_authors):
    print(i)
    file_loader = FileLoader(base_path + '/' + str(i + 1) + '.txt')
    files.append(file_loader.load_file()[0])

print(files[0])
print(files[1])
print(files[2])

for i in range(number_of_authors):
    # TODO
    #
    #
    #
    #  Preprocessing
    pass


# for target in folders['problems']:
#     corpus_1 = TextProcess()
#     corpus_list.append(corpus_1.get_data(base_path + '/' + target + '/known01.txt'))
#     corpus_2 = TextProcess()
#     corpus_list.append(corpus_2.get_data(base_path + '/' + target + '/unknown.txt'))
#
# print(corpus_list)

# num_batches = rep_tensor.shape[1] // timesteps
#
#
#
#
# model = TextGenerator(vocab_size, embed_size, hidden_size, num_layers)
# loss_fn = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#
# for epoch in range(num_epochs):
#     states = (torch.zeros(num_layers, batch_size, hidden_size),
#               torch.zeros(num_layers, batch_size, hidden_size))
#
#     for i in range(0, rep_tensor.size(1) - timesteps, timesteps):
#         inputs = rep_tensor[:, i:i + timesteps]
#         targets = rep_tensor[:, (i + 1):(i + 1) + timesteps]
#         outputs, _ = model(inputs, states)
#         loss = loss_fn(outputs, targets.reshape(-1))
#
#         model.zero_grad()
#         loss.backward()
#         clip_grad_norm(model.parameters(), 0.5)
#         optimizer.step()
#
#         step = (i + 1) // timesteps
#         if step % 1 == 0:
#             print('Epoch [{}/{}], Loss: {:4f}'.format(epoch + 1, num_epochs, loss.item()))
#
# with torch.no_grad():
#     with open('results.txt', 'w') as f:
#         state = (torch.zeros(num_layers, 1, hidden_size),
#                  torch.zeros(num_layers, 1, hidden_size))
#
#         input = torch.randint(0, vocab_size, (1,)).long().unsqueeze(1)
#
#         for i in range(500):
#             output, _ = model(input, state)
#             print(output.shape)
#
#             prob = output.exp()
#             word_id = torch.multinomial(prob, num_samples=1).item()
#             print(word_id)
#             input.fill_(word_id)
#
#             word = corpus.dictionary.idx2word[word_id]
#             word = '\n' if word == '<eos>' else word + ' '
#             f.write(word)
#
#             if (i + 1) % 100 == 0:
#                 print('Sampled [{}/{}] words and save to {}'.format(i + 1, 500, 'results.txt'))
