library(brms)


### Set working directory to be the folder in which the script resides
setwd(dirname(rstudioapi::getSourceEditorContext()$path))


### Load example data
data = read.table('data/data_m3_1.csv', sep=';', header=T)
N = nrow(data)

### Set priors over weights
prior0 = prior(normal(0, 1), class = "Intercept") 
priorb = prior(normal(0, 1), class = "Intercept") + prior(normal(0, 1), class = "b") 


### Model definitions
fml_m0 = bf(y ~ 1, sigma=1)
fml_m1 = y ~ 1 + x
fml_m2 = y ~ 1 + x + I(x^2)
fml_m3 = y ~ 1 + x + I(x^2) + I(x^3)


### Fit models
m0 = brm(fml_m0, data = data, warmup = 1000, prior = prior0,
         iter = 10000, cores = 4, chains = 4, seed = 42)
m0 = add_criterion(m0, "waic")
m0 = add_criterion(m0, "loo")

m1 = brm(fml_m1, data = data, warmup = 1000, prior = priorb,
         iter = 10000, cores = 4, chains = 4, seed = 42)
m1 = add_criterion(m1, "waic")
m1 = add_criterion(m1, "loo")

m2 = brm(fml_m2, data = data, warmup = 1000, prior = priorb,
         iter = 10000, cores = 4, chains = 4, seed = 42)
m2 = add_criterion(m2, "waic")
m2 = add_criterion(m2, "loo")

m3 = brm(fml_m3, data = data, warmup = 1000, prior = priorb,
         iter = 10000, cores = 4, chains = 4, seed = 42)
m3 = add_criterion(m3, "waic")
m3 = add_criterion(m3, "loo")
m3 = add_criterion(m3, "marg_l")
### Model comparison

# WAIC
waic_out = loo_compare(m0, m1, m2, m3, criterion = "waic")
print(waic_out, simplify=F)
# LOO
loo_compare(m0, m1, m2, m3, criterion = "loo")

# BF
bayes_factor(m0, m1)
bayes_factor(m0, m2)
bayes_factor(m0, m3)
bayes_factor(m1, m2)
bayes_factor(m1, m3)
bayes_factor(m2, m3)

# AIC
ml0 = max(rowSums(log_lik(m0, pointwise = F)))
ml1 = max(rowSums(log_lik(m1, pointwise = F)))
ml2 = max(rowSums(log_lik(m2, pointwise = F)))
ml3 = max(rowSums(log_lik(m3, pointwise = F)))

aic0 = 2 * 1 - 2 * ml0
aic1 = 2 * 2 - 2 * ml1
aic2 = 2 * 3 - 2 * ml2
aic3 = 2 * 3 - 2 * ml3

c(aic0, aic1, aic2, aic3)


# BIC
bic0 = log(N) * 1 - 2 * ml0
bic1 = log(N) * 2 - 2 * ml1
bic2 = log(N) * 3 - 2 * ml2
bic3 = log(N) * 3 - 2 * ml3

c(bic0, bic1, bic2, bic3)


# ALL
  