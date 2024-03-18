# Anisotropic covariance with aligned signal
set.seed(1)
n = 80; p = 500
rho = 0.2
Sig = rho^abs(outer(1:p, 1:p, "-")) # autoregressive

e = eigen(Sig)
b0 = e$vec[,1] / sqrt(e$val[1])
Sig.half = e$vec %*% diag(sqrt(e$val)) %*% t(e$vec)
lam.vec = exp(seq(log(1e-4), log(1e2), length = 30))

# Finite-sample bias and variance: ridge
Z = matrix(rnorm(n*p),n,p)
X = Z %*% Sig.half
Sig.hat = crossprod(X)/n
Id = diag(p)

b.emp = v.emp = numeric(length = length(lam.vec))
for (i in 1:length(lam.vec)) {
  cat(i, "... ")
  lam = lam.vec[i]
  Res = solve(Sig.hat + lam * Id)
  b.emp[i] = lam^2 * t(b0) %*% Res %*% Sig %*% Res %*% b0
  v.emp[i] = sum(diag(Sig.hat %*% Res %*% Res %*% Sig)) / n
}

# Finite-sample bias and variance: ridgeless
r = min(n, p)
e.hat = eigen(Sig.hat)
Sig.hat.pinv = e.hat$vec %*% diag(c(1/e.hat$val[1:r], rep(0, p-r))) %*% 
  t(e.hat$vec)
N = (Id - Sig.hat.pinv %*% Sig.hat)
b0.emp = t(b0) %*% N %*% Sig %*% N %*% b0 * (n < p)
v0.emp = sum(diag(Sig.hat.pinv %*% Sig)) / n
  
# Semi-asymptotic bias and variance: ridge
gam = p/n

# Which eigenvalues to use for the Silverstein equation?
# s = e$val # finite-sample 
N = 1000    
theta = 2*(1:N-1)*pi/(2*N+1)
s = (1-rho^2)/(1-2*rho*cos(theta)+rho^2)

# Function for numerically olving the Silverstein equation
silverstein = function(z, g, s,
                       interval = c(0, 50),
                       tol = .Machine$double.eps^0.25, 
                       maxiter = 1000) { 
  f = function(v, z, g, s) 1/v + z - g * mean(s/(1+s*v)) 
  root_obj = uniroot(f, interval, z, g, s,
                     tol = tol, maxiter = maxiter) 
  v = root_obj$root
  vp = 1/(1/v^2 - g * mean(s^2/(1+s*v)^2))
  return(list(v = v, vp = vp))
}

b.lim = v.lim = numeric(length = length(lam.vec))
for (i in 1:length(lam.vec)) {
  cat(i, "... ")
  lam = lam.vec[i]
  obj = silverstein(-lam, gam, s)
  v = obj$v; vp = obj$vp
  Res = solve(v * Sig + Id)
  b.lim[i] = vp / v^2 * t(b0) %*% Res %*% Sig %*% Res %*% b0
  v.lim[i] = vp / v^2 - 1
}

# Semi-asymptotic bias and variance: ridgeless
obj = silverstein(0, gam, s)
v = obj$v; vp = obj$vp
Res = solve(v * Sig + Id)
b0.lim = vp / v^2 * t(b0) %*% Res %*% Sig %*% Res %*% b0
v0.lim = vp / v^2 - 1

# Checks
max(abs(b.emp - b.lim))
max(abs(v.emp - v.lim))
abs(b0.emp - b0.lim)
abs(v0.emp - v0.lim)

# Plot things
par(mfrow = c(1, 2))
matplot(lam.vec, cbind(b.emp, b.lim), type = "l", lty = 1:2, col = 1:2)
matplot(lam.vec, cbind(v.emp, v.lim), type = "l", lty = 1:2, col = 1:2)
par(mfrow = c(1, 1))

alpha = 5
r.emp = alpha * b.emp + v.emp
r.lim = alpha * b.lim + v.lim
r0.emp = as.numeric(alpha * b0.emp + v0.emp)
r0.lim = as.numeric(alpha * b0.lim + v0.lim)

pdf(file = "ridge_ar.pdf", height = 5, width = 7)
par(mar = c(4.25, 4.25, 2.25, 1.25))
matplot(lam.vec, cbind(r.emp, r.lim), 
        type = "l", lty = 1:2, col = 1:2, 
        xlim = c(1e-5, 1e2), xaxt = "n", log = "x",
        main = "Autoregressive features, aligned signal",
        xlab = expression(lambda), ylab = "Risk")
points(1e-5, r0.emp, pch = 20, col = 1)
points(1e-5, r0.lim, pch = 21, cex = 0.7, col = 2)
axis(side = 1, at = 10^((-5):2), labels = c(0, 10^((-4):2)))
legend("topleft", c("Finite-sample", "Semi-asymptotic"), 
       lty = 1:2, col = 1:2)
graphics.off()

all(r0.emp < r.emp)
all(r0.lim < r.lim)
sum(b0^2) * alpha