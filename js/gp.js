"use strict";

var GP = { };

GP.kernels = [
  {
    name: 'squared exponential',
    hyperparams: [
      {'name': 'variance', default: 1.0, min: 0.05, max: 2.0, step: 0.05},
      {'name': 'length',   default: 0.2, min: 0.05, max: 1.0, step: 0.025},
      {'name': 'noise',    default: 0.1, min: 0.05, max: 1.0, step: 0.025}
    ],
    eval: function(x, y, theta) {
      return theta[0] * theta[0] * Math.exp(-Math.pow(x - y, 2) / (2 * theta[1] * theta[1])) + theta[2] * theta[2] * (x == y ? 1 : 0);
    }
  }, {
    name: 'exponential',
    hyperparams: [
      {'name': 'variance', default: 1.0, min: 0.05, max: 2.0, step: 0.05},
      {'name': 'length',   default: 0.2, min: 0.05, max: 1.0, step: 0.01},
      {'name': 'noise',    default: 0.1, min: 0.05, max: 1.0, step: 0.01}
    ],
    eval: function(x, y, theta) {
      return theta[0] * theta[0] * Math.exp(-Math.abs(x - y) / theta[1]) + theta[2] * theta[2] * (x == y ? 1 : 0);
    }
  }
];

var Regressor = function() {
  this.N = 0;
  this.kernelName = 'squared exponential';
};

Regressor.prototype.setKernel = function(name) {
  for (var i = 0; i < GP.kernels.length; ++i) {
    if (GP.kernels[i].name == name) {
      console.log('Setting kernel to', name);
      this.kernel = GP.kernels[i];
    }
  }
  this.hyperparams = { };
  for (var i = 0; i < this.kernel.hyperparams.length; ++i) {
    this.hyperparams[this.kernel.hyperparams[i].name] = this.kernel.hyperparams[i].default;
  }
};

Regressor.prototype.setData = function(x, y) {
  this.x = matrix(x);
  this.y = matrix(y);
  this.N = x.length;
};

Regressor.prototype.reset = function() {
  this.N = 0;
  this.evals = 0;
};

Regressor.prototype.getSigmaInplace = function(Sigma, theta) {
  for (var i = 0; i < this.N; ++i) {
    for (var j = i; j < this.N; ++j) {
      Sigma[i * this.N + j] = this.kernel.eval(this.x[i], this.x[j], theta);
      Sigma[j * this.N + i] = Sigma[i * this.N + j];
    }
  }
  return Sigma;
};

Regressor.prototype.getSigma = function(theta) {
  var Sigma = zeros(this.N, this.N);
  return this.getSigmaInplace(Sigma, theta);
};

Regressor.prototype.getHyperparams = function() {
  var theta = zeros(this.kernel.hyperparams.length);
  for (var i = 0; i < this.kernel.hyperparams.length; ++i) {
    theta[i] = this.hyperparams[this.kernel.hyperparams[i].name];
  }
  return theta;
};

Regressor.prototype.setHyperparams = function(theta) {
  for (var i = 0; i < this.kernel.hyperparams.length; ++i) {
    this.hyperparams[this.kernel.hyperparams[i].name] = theta[i];
  }
};

Regressor.prototype.getMoments = function(x_new) {
  var theta = this.getHyperparams();
  var K_new = zeros(x_new.length, this.N);
  for (var i = 0; i < x_new.length; ++i) {
    for (var j = 0; j < this.N; ++j) {
      K_new[i * this.N + j] = this.kernel.eval(x_new[i], this.x[j] + 1e-10, theta); // avoid adding dij
    }
  }
  var L = this.getSigma(theta).chol_inplace();
  var mean = K_new.multiply(L.bsolve(L.fsolve(this.y), {transpose: true}));
  var variance = zeros(x_new.length);
  for (var i = 0; i < x_new.length; ++i) {
    variance[i] = this.kernel.eval(x_new[i], x_new[i], theta) - Math.pow(L.fsolve_inplace(K_new.row(i)).norm(), 2);
  }
  return {mean: mean, variance: variance};
};

Regressor.prototype.logLikelihood = function(theta) {
  this.evals++;
  var L = this.getSigma(theta).chol_inplace();
  var logdet = L.diagonal().map(Math.log).sum();
  return -0.5 * this.N * Math.log(2 * Math.PI) - logdet - 0.5 * Math.pow(L.fsolve(this.y).norm(), 2);
};

Regressor.prototype.getRealizations = function(xstar) {
  var theta = this.getHyperparams();
  var KstarT = zeros(xstar.length, this.N);
  for (var i = 0; i < xstar.length; ++i) {
    for (var j = 0; j < this.N; ++j) {
      KstarT[i * this.N + j] = this.kernel.eval(xstar[i], this.x[j] + 1e-10, theta); // avoid adding dij
    }
  }
  var L = this.getSigma(theta).chol_inplace();
  var mean = KstarT.multiply(L.bsolve(L.fsolve(this.y), {transpose: true}));
  var covar = zeros(xstar.length, xstar.length);
  for (var i = 0; i < xstar.length; ++i) {
    for (var j = 0; j <  xstar.length; ++j) {
      covar[i * xstar.length + j] = this.kernel.eval(xstar[i], xstar[j] + 1e-10, theta);
    }
  }
  covar.increment(eye(xstar.length, xstar.length).scale(1e-8));
  var Kstar = KstarT.transpose();
  var KinvKstar = zeros(this.N, xstar.length);
  for (var j = 0; j < xstar.length; ++j) {
    KinvKstar.setCol(j, L.bsolve(L.fsolve(Kstar.col(j)), {transpose: true}));
  }
  covar = covar.subtract(KstarT.multiply(KinvKstar));
  var L = covar.chol();

  var getNormal = function() {
    var x, y, w;
    do {
      x = Math.random() * 2 - 1;
      y = Math.random() * 2 - 1;
      w = x * x + y * y;
    } while (w >= 1.0)
    return x * Math.sqrt(-2 * Math.log(w) / w);
  };

  var realizations = [ ];
  for (var i = 0; i < 5; ++i) {
    var z = Float64Array.build(getNormal, xstar.length, 1);
    realizations.push(mean.add(L.multiply(z)));
  }
  self.realizations = realizations;
  return realizations;
};

// Evaluate gradient w.r.t theta using forward difference
Regressor.prototype.gradientFD = function(theta) {
  var n = this.kernel.hyperparams.length;
  var h = 1e-6;
  var delta = eye(n).scale(h);
  var logLikelihood = this.logLikelihood(theta);
  var grad = zeros(n);
  for (var i = 0; i < n; ++i) {
    grad[i] = (this.logLikelihood(theta.add(delta.col(i))) - logLikelihood) / h;
  }
  return grad.scale(-1);
};

// Multistart BFGS
Regressor.prototype.optimize = function() {
  this.evals = 0;
  var theta = this.getHyperparams();
  var self = this;
  var proposals = [theta];
  for (var i = 0; i < 10; ++i) {
    var random_proposl = Float64Array.build(function(i) {
      return Math.random() * (self.kernel.hyperparams[i].max - self.kernel.hyperparams[i].min) + self.kernel.hyperparams[i].min;
    }, self.kernel.hyperparams.length);
    proposals.push(random_proposl);
  }
  var bestProposal = proposals[0].copy();
  var bestProposalValue = self.logLikelihood(bestProposal);
  for (var i = 0; i < proposals.length; ++i) {
    var proposal = proposals[i].copy();
    for (var j = 0; j < 5; ++j) {
      var result = BFGS(function(x) { return self.gradientFD(x); }, proposal, 10);
      proposal = result.optimum;
      if (result.errors.length > 0)
        break;
    }
    var value = self.logLikelihood(proposal);
    if (value > bestProposalValue) {
      bestProposalValue = value;
      bestProposal = proposal.copy();
    }
  }
  console.log(this.evals);
  this.setHyperparams(bestProposal);
  return proposal;
};
