import predict_all_core
from net import MyChainLSTM, MyChainBiDirectionalLSTM


def main():
    model = MyChainBiDirectionalLSTM(in_size=9, hidden_size=5, seq_size=20, out_size=3,
                        device=-1, lstm_do_ratio=0.1, affine_do_ratio=0.5)
    predict_all_core.run(model)


if __name__ == '__main__':
    main()
