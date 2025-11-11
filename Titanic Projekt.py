import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 1) Titanic-Datensatz laden
# Aufgabe 1: Titanic-Dataset von seaborn
df_sns = sns.load_dataset("titanic")

# Beispiel: neue Spalte "new" (nicht unbedingt sinnvoll)
df_sns["new"] = df_sns["age"] / len(df_sns)

# Aufgabe 2: Titanic-Dataset aus CSV laden
csv_path = "C:/Users/homam/Downloads/archive/Titanic-Dataset.csv"
df = pd.read_csv(csv_path)

# 2) Erste Info
print("Missing values each column :")
print(df.isnull().sum())
print("Number of rows:", len(df))
print("/n")

# 3) Altersdurchschnitt berechnen
if "Age" in df.columns:
    average_age = df["Age"].mean()
    print("Average age:", average_age)
    df["Age"].fillna(average_age, inplace=True)
else:
    print("⚠️ 'Age'-Spalte nicht gefunden – wird übersprungen.")

# 4) Überprüfung der Wertehäufigkeit
print("Distributions (normalized):")
try:
    print(df.value_counts(normalize=True))
except Exception as e:
    print("Konnte Verteilungen nicht berechnen:", e)
print("//n")

# 5) Spalten aufräumen
# Cabin-Spalte entfernen (hat viele fehlende Werte)
if "Cabin" in df.columns:
    df = df.drop("Cabin", axis=1)

# Fehlende Werte im Alter mit Modus füllen
if "Age" in df.columns:
    df["Age"].fillna(df["Age"].mode()[0], inplace=True)

# Familiengröße berechnen
if "SibSp" in df.columns and "Parch" in df.columns:
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

# 6) Ausgabe
print(df.to_string())  # Nur Kopf zeigen, um Überlauf zu vermeiden

# 7) Berechnen der Überlebenden
print("///////////////////")#

def survival_rate(df, name):
    if name in df.columns and "Survived" in df.columns:
        result = (
            df.groupby(name)["Survived"]
            .mean()
            .mul(100)
            .round(2)
            .sort_values(ascending=False)
        )
        return result
    else:
        print(f"⚠️ Spalte '{name}' oder 'Survived' fehlt.")
        return pd.Series()

if "Sex" in df.columns:
    result = survival_rate(df, "Sex")
    print(result)
print("///////////////")#

def age(df, name):
    if name in df.columns:
        summary = df[name].describe(percentiles=[0.25, 0.5, 0.75]).to_dict()
        return summary
    else:
        print(f"⚠️ Spalte '{name}' nicht gefunden.")
        return {}

if "Age" in df.columns:
    summary = age(df, "Age")
    print(summary)

# Class
class Passenger:
    def __init__(self, sex, age, pclass, fare, embarked, family_size, survived):
        self.sex = sex
        self.age = age
        self.pclass = pclass
        self.fare = fare
        self.embarked = embarked
        self.family_size = family_size
        self.survived = survived

    def is_child(self):
        return "Yes" if self.age < 18 else "No"

    def summary(self):
        if self.survived == 1:
            return "Alive"
        else:
            return ("Dead", self.sex, self.age, self.pclass)


# Beispielpassagiere
p1 = Passenger(sex='Male', age=40, pclass=92, fare=2, embarked='sdd', family_size=4, survived=1)
p2 = Passenger('Male', 35, 521, 4, 'dsdf', 2, 0)
print(p1.summary())


# TitanicAnalyzer-Class


class TitanicAnalyzer:
    def __init__(self, df):
        self.df = df.copy()

    def top_fares(self):
        if "Fare" in self.df.columns:
            return self.df["Fare"].sort_values(ascending=False, ignore_index=True).head()
        else:
            return pd.Series(dtype=float)

    def survival_by(self, column):
        return survival_rate(self.df, column)

    def age_stats(self, name):
        return age(self.df, name)

    def plot_survival(self, column):
        if column in self.df.columns:
            self.df[column].value_counts().plot(kind='bar')
            plt.show()
        else:
            print(f" Spalte '{column}' nicht vorhanden.")

    def Oreport(self,name):
        result = (self.df.groupby(name)["Survived"].mean().mul(100).round(2).sort_values(ascending=False))
        result2  = self.df["Age"].mean()
        print("This is the final report")
        print("*********************")
        print(f"The average age is {result2}: ")
        print(f"The average Survival is {result}: ")
        print("The End")




h = TitanicAnalyzer(df)
h.plot_survival("Survived")
h.age_stats("Age")
h.Oreport("Survived")
print("////////////////")
if "Sex" in df.columns:
    print(h.survival_by("Sex"))


# Passagier-Ausgabe

if all(x in df.columns for x in ["Sex", "Age", "Survived"]):
    for i in range(min(10, len(df))):
        passenger = df.iloc[i]
        print(f"Passenger{i+1}: Sex={passenger['Sex']}, Age={passenger['Age']}, Survived={passenger['Survived']}")

print("################")

i = 0
while i < min(5, len(df)):
    passenger = df.iloc[i]
    print(f"Passenger{i+1}: Sex={passenger['Sex']}, Age={passenger['Age']}, Survived={passenger['Survived']}")
    i += 1

# Diagramme (Fehlerfrei angepasst)
if "Survived" in df.columns:
    ax = df["Survived"].value_counts().plot.bar()
    ax.set_xticklabels(["Died", "Survived"])
    plt.xlabel("Status")
    plt.ylabel("Number of Passengers")
    plt.title("Titanic Survival")
    plt.show()

if "Age" in df.columns:
    plt.hist(df["Age"], bins=30, density=True)
    plt.ylabel("Probability")
    plt.xlabel("Age")
    plt.show()

if "Survived" in df.columns:
    sns.countplot(x="Survived", data=df)
if "Fare" in df.columns:
    sns.boxplot(x=df["Fare"])
if all(x in df.columns for x in ["Survived", "Age"]):
    sns.violinplot(data=df, x="Survived", y="Age")

if all(x in df.columns for x in ["Sex", "Pclass", "Survived"]):
    titanic = df.groupby(["Sex", "Pclass"])["Survived"].mean().unstack()
    print(titanic)

num_cols = [c for c in ["Age", "SibSp", "Parch", "Fare", "FamilySize"] if c in df.columns]
if len(num_cols) > 1:
    corr = df[num_cols].corr()
    sns.heatmap(corr, annot=True)
    plt.show()

#Homam
#Test
# Überlebensrate berechnen

def survival_rate2(df, name):
    if name in df.columns:
        survived = (df[name].astype("int").mean() * 100).round(2)
        return survived
    else:
        print(f" Spalte '{name}' fehlt.")
        return np.nan

if "Survived" in df.columns:
    print(survival_rate2(df, "Survived"), "%")
