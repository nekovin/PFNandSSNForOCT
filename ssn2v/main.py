from ssn2v.stage2.run import stage2

def main():
    stage2(
        train = False, test = False, evaluate=True
        )

if __name__ == "__main__":
    main()