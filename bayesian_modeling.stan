data {
  int<lower=1> N;             // number of observations
  int<lower=1> N_forecast;    // number of days to forecast
  int<lower=1> K_dow;         // number of day-of-week categories
  int<lower=1,upper=K_dow> dow[N];    // day-of-week index for historical data
  int<lower=1,upper=K_dow> dow_forecast[N_forecast]; // day-of-week index for forecasts
  int<lower=0> y[N];          // observed counts
  real t[N];                  // scaled time index
  real t_forecast[N_forecast]; // scaled future time points
  int<lower=1> S;             // number of Fourier terms
  matrix[N, 2 * S] fourier_terms; // Fourier terms for historical data
  matrix[N_forecast, 2 * S] fourier_terms_forecast; // Fourier terms for forecast
}


parameters {
  real alpha;                         // intercept
  vector[K_dow-1] beta_raw;           // day-of-week effects (except baseline)
  real delta;                         // trend coefficient
  vector[2 * S] gamma;                // Fourier coefficients
}


transformed parameters {
  vector[K_dow] beta;
  beta[1] = 0; // baseline day of week (e.g., Monday)
  for (i in 2:K_dow) {
    beta[i] = beta_raw[i-1];
  }
}


model {
  // Priors
  alpha ~ normal(0, 5);
  beta_raw ~ normal(0, 2);
  delta ~ normal(0, 2); //was 1, changed to 2
  gamma ~ normal(0, 1); // Prior for Fourier coefficients, was 0.5, changed to 1 
  // Likelihood
  for (n in 1:N) {
    real log_lambda = alpha + beta[dow[n]] + delta * t[n] + dot_product(gamma, fourier_terms[n]);
    target += poisson_log_lpmf(y[n] | log_lambda);
  }
}


generated quantities {
  vector[N_forecast] y_pred;
  for (f in 1:N_forecast) {
    real log_lambda_forecast = alpha + beta[dow_forecast[f]] + delta * t_forecast[f] + dot_product(gamma, fourier_terms_forecast[f]);
    y_pred[f] = poisson_rng(exp(log_lambda_forecast));
  }
}
