# References

This directory contains reference materials for the project:

## Data Dictionaries

### Obesity Estimation Dataset

| Variable | Description | Type | Values/Range | Notes |
|----------|-------------|------|--------------|-------|
| Gender | Género del individuo | Categorical | Male, Female | - |
| Age | Edad en años | Numeric | 5-50 | Rango válido según dataset |
| Height | Altura en metros | Numeric | 0.5-2.5 | Validación de rango físico |
| Weight | Peso en kilogramos | Numeric | 20-300 | Rango estimado para validación |
| family_history_with_overweight | Historial familiar con sobrepeso | Categorical | yes, no | Factor de riesgo importante |
| FAVC | Consumo frecuente de alimentos altos en calorías | Categorical | yes, no | Hábito alimenticio |
| FCVC | Frecuencia de consumo de vegetales | Numeric | 1-3 | Escala: 1=raramente, 3=siempre |
| NCP | Número de comidas principales | Numeric | 1-4 | Comidas al día |
| CAEC | Consumo de alimentos entre comidas | Categorical | no, Sometimes, Frequently, Always | Snacking |
| SMOKE | Fumador | Categorical | yes, no | Factor de estilo de vida |
| CH2O | Consumo diario de agua en litros | Numeric | 1-3 | Hidratación |
| SCC | Monitoreo de consumo de calorías | Categorical | yes, no | Autocontrol |
| FAF | Frecuencia de actividad física | Numeric | 0-3 | Escala: 0=nunca, 3=diario |
| TUE | Tiempo de uso de dispositivos tecnológicos | Numeric | 0-2 | Horas al día |
| CALC | Consumo de alcohol | Categorical | no, Sometimes, Frequently, Always | Hábito |
| MTRANS | Medio de transporte usado | Categorical | Public_Transportation, Walking, Automobile, Motorbike, Bike | Actividad física indirecta |
| NObeyesdad | Nivel de obesidad (Target) | Categorical | Insufficient_Weight, Normal_Weight, Overweight_Level_I, Overweight_Level_II, Obesity_Type_I, Obesity_Type_II, Obesity_Type_III | Variable objetivo (7 clases) |

## External References

### Organizaciones de Salud
- [WHO BMI Classification](https://www.who.int/news-room/fact-sheets/detail/obesity-and-overweight) - Clasificación de obesidad según OMS
- [CDC - Adult Obesity Facts](https://www.cdc.gov/obesity/data/adult.html) - Estadísticas de obesidad

### Herramientas y Frameworks
- [Cookiecutter Data Science](https://cookiecutter-data-science.drivendata.org/) - Estructura de proyecto
- [DVC Documentation](https://dvc.org/doc) - Versionado de datos
- [MLflow Documentation](https://mlflow.org/docs/latest/) - Experiment tracking
- [FastAPI Documentation](https://fastapi.tiangolo.com/) - Framework API REST
- [XGBoost Documentation](https://xgboost.readthedocs.io/) - Modelo de ML utilizado

## Papers and Articles

- Dataset original disponible en repositorios públicos de Machine Learning
- Referencias sobre clasificación de obesidad según estándares internacionales (OMS)
