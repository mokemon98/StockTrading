import simulate_trading_cross
from net import MyChain, MyChainLSTM, MyChainBiDirectionalLSTM


def main():
    model = MyChainBiDirectionalLSTM(in_size=9, hidden_size=5, seq_size=20)
    simulate_trading_cross.run(model)


if __name__ == '__main__':
    main()
