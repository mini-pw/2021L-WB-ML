#setwd("~/Documents/Sem6/WB/app/")
library(readr)
library(DALEX)
library(shiny)
library("ranger")
setwd("C:/Users/hrucz/Dropbox/WB_Faza2")
#setwd("C:/Users/Kinga/Documents/Dropbox/WB_Faza2")
covid <- read_csv("200518COVID19MEXICO.csv")

covid=covid[1:150200, c(6,14,15,16,20,21,22,23,24,26,27,28,29,30,31)]
covid
#SEXO, INTUBADO, NEUMONIA, EDAD, DIABETES, EPOC, ASMA, INMUSUPR, HIPERTENSION, nie ma-OTRA_COM, CARDIOVASCULAR,OBESIDAD
#RENAL_CRONICA, TABAQUISMO, OTRO_CASO, RESULTADO

set.seed(123)
covid10 = covid[sample(nrow(covid),10000),]
ranger_model <- ranger::ranger(RESULTADO~., data = covid10, classification = TRUE, probability = TRUE)
custom_predict <- function(X.model, new_data) {
  predict(X.model, new_data)$predictions[,2]
}

explainer = explain(model = ranger_model, data = covid10[, -15], label = "Random Forest")
?explain
ui <- fluidPage(
  
  titlePanel("Covid-19 Predict"),
  
  sidebarLayout(
    sidebarPanel(
      helpText("Please choose the variables that describe your person to predict the risk."),
      helpText("Mind that calculating an outcome might take a while"),
      selectInput(inputId = "Sex",label =  "Choose your sex",
                  choices = c("Female", "Male"),
                  selected = 'Male'),
      
      sliderInput(inputId = "Age",
                  label = "Choose your age",
                  min = 1,
                  max = 100,
                  value = 18),
      
      selectInput(inputId = "Intubation",label =  "Did you have Intubation?",
                  choices = c("Yes", "No"),
                  selected='No'),
      
      selectInput(inputId = "Pneumonia",label =  "Do you have Pneumonia?",
                  choices = c("Yes", "No"),
                  selected='No'),
      
      selectInput(inputId = "Diabetes",label =  "Do you have Diabetes?",
                  choices = c("Yes", "No"),
                  selected='No'),
      
      selectInput(inputId = "Asthma",label =  "Do you have Asthma?",
                  choices = c("Yes", "No"),
                  selected='No'),
      
      selectInput(inputId = "Hypertension",label =  "Do you have Hypertension?",
                  choices = c("Yes", "No"),
                  selected='No'),
      
      #selectInput(inputId = "Otro_Com",label =  "Do you have otro com?",
      #            choices = c("Yes", "No"),
      #            selected='No'),
      
      selectInput(inputId = "Obesity",label =  "Are you obese?",
                  choices = c("Yes", "No"),
                  selected='No'),
      
      selectInput(inputId = "Cardio_vascular_diseases",label =  "Do you have any Cardio-vascular diseases?",
                  choices = c("Yes", "No"),
                  selected='No'),
      
      selectInput(inputId = "Smoking",label =  "Do you smoke?",
                  choices = c("Yes", "No"),
                  selected='No'),
      
      selectInput(inputId = "epoc",label =  "Do you have Chronic Obstructive Pulmonary Disease(EPOC in our data)?",
                  choices = c("Yes", "No"),
                  selected='No'),
      
      selectInput(inputId = "Immunosuppression",label =  "Do you have Immunosuppression?",
                  choices = c("Yes", "No"),
                  selected='No'),
      
      selectInput(inputId = "Chronic_kidney_failure",label =  "Do you have Chronic kidney failure?",
                  choices = c("Yes", "No"),
                  selected='No'),
      
      selectInput(inputId = "Other",label =  "Do you have aby other chronic disease?", #otro_caso
                  choices = c("Yes", "No"),
                  selected='No')
    ),
    
    mainPanel(
      tabsetPanel(
        tabPanel("Your Covid Risk",
                 textOutput("covid_risk"),
                 helpText("The visualization below represents how most important parameters are affecting the outcome."),
                 helpText("Every number on this plots represents how much does this variable matter in spite of probablility of getting ill. The higher the number of
                          Covid risk the higher probablilty to get ill."),
                 helpText("On this plot green colour shows that the factor makes getting ill more likely and red represents the opposite."),
                 plotOutput(outputId = "breakdown")
                 
                 
        ),
        
        tabPanel("Profile plot",
                 helpText("This plot illustrates the meaning of age given the selected conditions."),
                 helpText("This tool might be useful for providing information about the chance of getting Covid-19 varying on the age"),
                 plotOutput(outputId = "profilePlot")
                 
        ),
        tabPanel("Shap",
                 helpText("Shap analysis provides visualisation of the most important varaibles for the outcome and presents how do they look like in comparison to the others."),
                 helpText("On this plot green colour shows that the factor makes getting ill more likely and red represents the opposite."),
                 plotOutput(outputId = "shap")
        )
                 
      )
    )
  )
)

server <- function(input, output) {
  
  output$profilePlot <- renderPlot({
    
    EDAD <- input$Age
    if (input$Sex == 'Male'){SEXO <- 2}else{SEXO <- 1}
    if (input$Intubation == 'Yes'){INTUBADO <- 1}else{INTUBADO <- 2}
    if (input$Pneumonia == 'Yes'){NEUMONIA<- 1}else{NEUMONIA <- 2}
    if (input$Diabetes == 'Yes'){DIABETES<- 1}else{DIABETES <- 2}
    if (input$Asthma == 'Yes'){ASMA <- 1}else{ASMA <- 2}
    if (input$Hypertension == 'Yes'){HIPERTENSION <- 1}else{HIPERTENSION <- 2}
    #if (input$Otro_Com == 'Yes'){OTRA_COM<- 1}else{OTRA_COM <- 2}
    if (input$Obesity == 'Yes'){OBESIDAD <- 1}else{OBESIDAD <- 2}
    if (input$Cardio_vascular_diseases == 'Yes'){CARDIOVASCULAR <- 1}else{CARDIOVASCULAR <- 2}
    if (input$Smoking == 'Yes'){TABAQUISMO <- 1}else{TABAQUISMO <- 2}
    if (input$epoc == 'Yes'){EPOC <- 1}else{EPOC <- 2}
    if (input$Immunosuppression == 'Yes'){INMUSUPR <- 1}else{INMUSUPR <- 2}
    if (input$Chronic_kidney_failure == 'Yes'){RENAL_CRONICA <- 1}else{RENAL_CRONICA <- 2}
    if (input$Other == 'Yes'){OTRO_CASO <- 1}else{OTRO_CASO <- 2}
    input_data <- data.frame(SEXO, INTUBADO, NEUMONIA, EDAD, DIABETES, EPOC, ASMA, INMUSUPR, HIPERTENSION, CARDIOVASCULAR,
                             OBESIDAD, RENAL_CRONICA, TABAQUISMO, OTRO_CASO)
    
    profile <- individual_profile(explainer, new_observation =  input_data, variables = "EDAD")
    plot(profile, variables = "EDAD")
  })
  
  output$breakdown <- renderPlot({
    EDAD <- input$Age
    if (input$Sex == 'Male'){SEXO <- 2}else{SEXO <- 1}
    if (input$Intubation == 'Yes'){INTUBADO <- 1}else{INTUBADO <- 2}
    if (input$Pneumonia == 'Yes'){NEUMONIA<- 1}else{NEUMONIA <- 2}
    if (input$Diabetes == 'Yes'){DIABETES<- 1}else{DIABETES <- 2}
    if (input$Asthma == 'Yes'){ASMA <- 1}else{ASMA <- 2}
    if (input$Hypertension == 'Yes'){HIPERTENSION <- 1}else{HIPERTENSION <- 2}
    #if (input$Otro_Com == 'Yes'){OTRA_COM<- 1}else{OTRA_COM <- 2}
    if (input$Obesity == 'Yes'){OBESIDAD <- 1}else{OBESIDAD <- 2}
    if (input$Cardio_vascular_diseases == 'Yes'){CARDIOVASCULAR <- 1}else{CARDIOVASCULAR <- 2}
    if (input$Smoking == 'Yes'){TABAQUISMO <- 1}else{TABAQUISMO <- 2}
    if (input$epoc == 'Yes'){EPOC <- 1}else{EPOC <- 2}
    if (input$Immunosuppression == 'Yes'){INMUSUPR <- 1}else{INMUSUPR <- 2}
    if (input$Chronic_kidney_failure == 'Yes'){RENAL_CRONICA <- 1}else{RENAL_CRONICA <- 2}
    if (input$Other == 'Yes'){OTRO_CASO <- 1}else{OTRO_CASO <- 2}
    input_data <- data.frame(SEXO, INTUBADO, NEUMONIA, EDAD, DIABETES, EPOC, ASMA, INMUSUPR, HIPERTENSION, CARDIOVASCULAR,
                             OBESIDAD, RENAL_CRONICA, TABAQUISMO, OTRO_CASO)
    bd_ranger <- predict_parts_break_down(explainer, new_observation = input_data)
    plot(bd_ranger)
  })
  
  output$shap <- renderPlot({
    EDAD <- input$Age
    if (input$Sex == 'Male'){SEXO <- 2}else{SEXO <- 1}
    if (input$Intubation == 'Yes'){INTUBADO <- 1}else{INTUBADO <- 2}
    if (input$Pneumonia == 'Yes'){NEUMONIA<- 1}else{NEUMONIA <- 2}
    if (input$Diabetes == 'Yes'){DIABETES<- 1}else{DIABETES <- 2}
    if (input$Asthma == 'Yes'){ASMA <- 1}else{ASMA <- 2}
    if (input$Hypertension == 'Yes'){HIPERTENSION <- 1}else{HIPERTENSION <- 2}
    #if (input$Otro_Com == 'Yes'){OTRA_COM<- 1}else{OTRA_COM <- 2}
    if (input$Obesity == 'Yes'){OBESIDAD <- 1}else{OBESIDAD <- 2}
    if (input$Cardio_vascular_diseases == 'Yes'){CARDIOVASCULAR <- 1}else{CARDIOVASCULAR <- 2}
    if (input$Smoking == 'Yes'){TABAQUISMO <- 1}else{TABAQUISMO <- 2}
    if (input$epoc == 'Yes'){EPOC <- 1}else{EPOC <- 2}
    if (input$Immunosuppression == 'Yes'){INMUSUPR <- 1}else{INMUSUPR <- 2}
    if (input$Chronic_kidney_failure == 'Yes'){RENAL_CRONICA <- 1}else{RENAL_CRONICA <- 2}
    if (input$Other == 'Yes'){OTRO_CASO <- 1}else{OTRO_CASO <- 2}
    input_data <- data.frame(SEXO, INTUBADO, NEUMONIA, EDAD, DIABETES, EPOC, ASMA, INMUSUPR, HIPERTENSION, CARDIOVASCULAR,
                             OBESIDAD, RENAL_CRONICA, TABAQUISMO, OTRO_CASO)
    
    shap_ranger <- predict_parts_shap(explainer, new_observation = input_data)
    plot(shap_ranger)
  })
  
  output$covid_risk <- renderText({ 
    
    EDAD <- input$Age
    if (input$Sex == 'Male'){SEXO <- 2}else{SEXO <- 1}
    if (input$Intubation == 'Yes'){INTUBADO <- 1}else{INTUBADO <- 2}
    if (input$Pneumonia == 'Yes'){NEUMONIA<- 1}else{NEUMONIA <- 2}
    if (input$Diabetes == 'Yes'){DIABETES<- 1}else{DIABETES <- 2}
    if (input$Asthma == 'Yes'){ASMA <- 1}else{ASMA <- 2}
    if (input$Hypertension == 'Yes'){HIPERTENSION <- 1}else{HIPERTENSION <- 2}
    #if (input$Otro_Com == 'Yes'){OTRA_COM<- 1}else{OTRA_COM <- 2}
    if (input$Obesity == 'Yes'){OBESIDAD <- 1}else{OBESIDAD <- 2}
    if (input$Cardio_vascular_diseases == 'Yes'){CARDIOVASCULAR <- 1}else{CARDIOVASCULAR <- 2}
    if (input$Smoking == 'Yes'){TABAQUISMO <- 1}else{TABAQUISMO <- 2}
    if (input$epoc == 'Yes'){EPOC <- 1}else{EPOC <- 2}
    if (input$Immunosuppression == 'Yes'){INMUSUPR <- 1}else{INMUSUPR <- 2}
    if (input$Chronic_kidney_failure == 'Yes'){RENAL_CRONICA <- 1}else{RENAL_CRONICA <- 2}
    if (input$Other == 'Yes'){OTRO_CASO <- 1}else{OTRO_CASO <- 2}
    input_data <- data.frame(SEXO, INTUBADO, NEUMONIA, EDAD, DIABETES, EPOC, ASMA, INMUSUPR, HIPERTENSION, CARDIOVASCULAR,
                             OBESIDAD, RENAL_CRONICA, TABAQUISMO, OTRO_CASO)
    risk <- custom_predict(ranger_model, input_data)
    
    paste("Your Covid risk equals: ", risk*100, "%")
  })
  
  
}


shinyApp(ui = ui, server = server)