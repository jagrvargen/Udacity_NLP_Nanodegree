from keras import backend as K
from keras.models import Model
from keras.layers import (BatchNormalization, Conv1D, Dense, Input, 
    TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM, MaxPooling1D)

def simple_rnn_model(input_dim, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(output_dim, return_sequences=True, 
                 implementation=2, name='rnn')(input_data)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(simp_rnn)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def rnn_model(input_dim, units, activation, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(units, activation=activation,
        return_sequences=True, implementation=2, name='rnn')(input_data)
    # TODO: Add batch normalization 
    bn_rnn = BatchNormalization(name='bn_rnn_1')(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def cnn_rnn_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    # Add a recurrent layer
    simp_rnn = SimpleRNN(units, activation='relu',
        return_sequences=True, name='rnn')(bn_cnn)
    # TODO: Add batch normalization
    bn_rnn = BatchNormalization(name='bn_rnn_norm_1')(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def cnn_output_length(input_length, filter_size, border_mode, stride,
                       dilation=2):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // stride

def deep_rnn_model(input_dim, units, recur_layers, output_dim=29):
    """ Build a deep recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add recurrent layers, each with batch normalization
    gru_rnn_1 = GRU(units, activation='relu', return_sequences=True, name='rnn_1')(input_data)
    bn_rnn_1 = BatchNormalization(name='bn_1')(gru_rnn_1)

    # Add subsequent GRU layers if recur_layers > 1
    if recur_layers > 1:
        layer_index = [i + 1 for i in range(1, recur_layers)]
        layer_dict = {'bn_rnn_1': bn_rnn_1}
        for index in layer_index:
            layer_dict['gru_rnn_{}'.format(str(index))] = GRU(units, activation='relu',
                                                   return_sequences=True,
                                                   name=('rnn_' + str(index)))(layer_dict['bn_rnn_{}'.format(str(index - 1))])
            layer_dict['bn_rnn_{}'.format(str(index))] = BatchNormalization(name=('bn_' + str(index)))(layer_dict['gru_rnn_{}'.format(str(index))])
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    if recur_layers > 1:
        time_dense = TimeDistributed(Dense(output_dim))(layer_dict['bn_rnn_{}'.format(str(layer_index[-1]))])
    else:
        time_dense = TimeDistributed(Dense(output_dim))(bn_rnn_1)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def bidirectional_rnn_model(input_dim, units, output_dim=29):
    """ Build a bidirectional recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add bidirectional recurrent layer
    bidir_rnn = Bidirectional(LSTM(units, return_sequences=True, name='bi_rnn'))(input_data)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bidir_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def final_model(input_dim, units, recur_layers, dilation, filters,
                kernel_size, conv_stride, conv_border_mode, output_dim=29, dropout=0.2):
    """ Build a deep network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))

    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     dilation_rate=dilation,
                     activation='relu',
                     name='conv1d')(input_data)
#    max_pool = MaxPooling1D(pool_size=2, strides=1, padding='valid')(conv_1d)
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    
    # Add bidirectional RNN layers
    bidir_rnn_1 = Bidirectional(GRU(units, dropout=dropout, recurrent_dropout=dropout, activation='relu',
                                    return_sequences=True, name='bi_rnn'))(bn_cnn)
    bn_rnn_1 = BatchNormalization(name='bn_1')(bidir_rnn_1)
    
    if recur_layers > 1:
        layer_index = [i + 1 for i in range(1, recur_layers)]
        layer_dict = {'bn_rnn_1': bn_rnn_1}
        for index in layer_index:
            layer_dict['bidir_rnn_{}'.format(str(index))] = Bidirectional(GRU(units, dropout=dropout,
                                                                             recurrent_dropout=dropout,
                                                                             activation='relu',
                                                                             return_sequences=True,
                                                                             name=('rnn_' + str(index))))(layer_dict['bn_rnn_{}'.format(str(index - 1))])
            layer_dict['bn_rnn_{}'.format(str(index))] = BatchNormalization(name=('bn_' + str(index)))(layer_dict['bidir_rnn_{}'.format(str(index))])

    if recur_layers > 1:
        time_dense = TimeDistributed(Dense(output_dim))(layer_dict['bn_rnn_{}'.format(str(layer_index[-1]))])
    else:
        time_dense = TimeDistributed(Dense(output_dim))(bn_rnn_1)
    
    # TODO: Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    # TODO: Specify model.output_length
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model