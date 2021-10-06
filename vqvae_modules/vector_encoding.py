## Encoding vectors

#univariate
def dictionary_encoding(intermediate_output):

  global i
  global j
  i = -1
  j = -1

  encoded_vectors = {}
  for emb_1 in intermediate_output:
    i +=1
    j = -1
    for emb_2 in emb_1:
      j +=1
      emb_2 = tuple(emb_2)
      #print(emb_2)
      if emb_2 in encoded_vectors:
        previous = encoded_vectors[emb_2]
        previous.append((i,j))
        encoded_vectors[emb_2] = previous
      else:
        encoded_vectors[emb_2] = [(i,j)]
  return encoded_vectors

#multivariate

def encoding_n(output):

  global i
  global j
  i = -1
  j = -1
  k = -1

  encoded = {}
  for emb_1 in output:
    i += 1
    j = -1
    for emb_2 in emb_1:
      j +=1
      for emb_3 in emb_2:
        k +=1
        emb_3 = tuple(emb_3)
        if emb_3 in encoded:
          previous = encoded[emb_3]
          previous.append((i,j,k))
        else:
          encoded[emb_3] = (i,j,k)
  return encoded