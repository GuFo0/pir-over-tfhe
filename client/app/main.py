import string
import sys

from client.app.fhe_client import PIRClient


def main(port):
    print("=============================================")
    print("       TFHE PIR SPAM FILTER CLIENT           ")
    print("=============================================\n")
    client = PIRClient(server_url="http://localhost:" + port)

    try:
        client.initialize_session()
    except Exception as e:
        print(f"[ERRORE] Inizializzazione fallita: {e}")
        sys.exit(1)

    while True:
        print("---------------------------------------------")
        user_input = input(
            "Inserisci l'indice del numero da verificare (o 'q' per uscire): "
        )

        if user_input.lower() == "q":
            print("Chiusura applicazione client.")
            break

        try:
            phone_index = int(user_input)
            if phone_index < 0:
                print("[ATTENZIONE] Inserire un numero positivo.")
                continue
        except ValueError:
            print("[ATTENZIONE] Input non valido. Inserire un numero intero decimale.")
            continue

        try:
            is_spam = client.verify_phone_number(phone_index)

            print("\n>>>  RISULTATO  <<<")
            if is_spam:
                print(
                    f"L'indice {phone_index} è associato a un numero di SPAM! [RILEVATO]"
                )
            else:
                print(f"L'indice {phone_index} è sicuro. [NON RILEVATO]")
            print("---------------------------------------------\n")

        except Exception as e:
            print(f"[ERRORE] Impossibile completare la query: {e}\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Inserire la porta del server come parametro")
        exit(1)
    main(sys.argv[1])
