from MarIA import MariaRoberta


if __name__ == "__main__":
    MODEL_NAME = 'PlanTL-GOB-ES/roberta-base-bne'
    model = MariaRoberta(MODEL_NAME)

    print(model("¡Hola <mask>!"))
