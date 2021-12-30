import datasets
import model
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split
import unicodedata
import re
import numpy as np
import os
import io
import time

@tf.function
def train_step(inp, targ, enc_hidden):
  loss = 0

  with tf.GradientTape() as tape:
    enc_output, enc_hidden = model.encoder(inp, enc_hidden)

    dec_hidden = enc_hidden

    dec_input = tf.expand_dims([datasets.targ_lang.word_index['<start>']] * datasets.BATCH_SIZE, 1)

    # Teacher forcing - feeding the target as the next input
    for t in range(1, targ.shape[1]):
      # passing enc_output to the decoder
      predictions, dec_hidden, _ = model.decoder(dec_input, dec_hidden, enc_output)

      loss += model.loss_function(targ[:, t], predictions)

      # using teacher forcing
      dec_input = tf.expand_dims(targ[:, t], 1)

  batch_loss = (loss / int(targ.shape[1]))

  variables = model.encoder.trainable_variables + model.decoder.trainable_variables

  gradients = tape.gradient(loss, variables)

  model.optimizer.apply_gradients(zip(gradients, variables))

  return batch_loss

def main():
  EPOCHS = 2

  for epoch in range(EPOCHS):
    start = time.time()

    enc_hidden = model.encoder.initialize_hidden_state()
    total_loss = 0

    for (batch, (inp, targ)) in enumerate(datasets.dataset.take(datasets.steps_per_epoch)):
      batch_loss = train_step(inp, targ, enc_hidden)
      total_loss += batch_loss

      if batch % 100 == 0:
        print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                     batch,
                                                     batch_loss.numpy()))
    # saving (checkpoint) the model every 2 epochs
    if (epoch + 1) % 2 == 0:
      model.checkpoint.save(file_prefix=model.checkpoint_prefix)

    print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                        total_loss / datasets.steps_per_epoch))
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

if __name__ == "__main__":
            main()
