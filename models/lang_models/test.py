from maria import MariaRoberta
from politibeto import PolitiBeto


if __name__ == "__main__":
    MODEL_NAME = 'PlanTL-GOB-ES/roberta-base-bne'
    model = MariaRoberta(MODEL_NAME)

    print(model("¡Hola <mask>!"))

    model2 = PolitiBeto()
    print(model2("¡Hola <mask>!"))
