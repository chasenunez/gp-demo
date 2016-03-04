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
};

Regressor.prototype.computeSigmaInplace = function(Sigma, theta) {
  for (var i = 0; i < this.N; ++i) {
    for (var j = i; j < this.N; ++j) {
      Sigma[i * this.N + j] = this.kernel.eval(this.x[i], this.x[j], theta);
      Sigma[j * this.N + i] = Sigma[i * this.N + j];
    }
  }
  return Sigma;
};

Regressor.prototype.computeSigma = function(theta) {
  var Sigma = zeros(this.N, this.N);
  return this.computeSigmaInplace(Sigma, theta);
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
  var L = this.computeSigma(theta).chol_inplace();
  var mean = K_new.multiply(L.bsolve(L.fsolve(this.y), {transpose: true}));
  var variance = zeros(x_new.length);
  for (var i = 0; i < x_new.length; ++i) {
    variance[i] = this.kernel.eval(x_new[i], x_new[i], theta) - Math.pow(L.fsolve_inplace(K_new.row(i)).norm(), 2);
  }
  return {mean: mean, variance: variance};
};

Regressor.prototype.logLikelihood = function(theta) {
  var L = this.computeSigma(theta).chol_inplace();
  var logdet = L.diagonal().map(Math.log).sum();
  return -0.5 * this.N * Math.log(2 * Math.PI) - logdet - 0.5 * Math.pow(L.fsolve(this.y).norm(), 2);
};

Regressor.prototype.gradientFD = function(theta) {
  var n = this.kernel.hyperparams.length;
  var h = 1e-8;
  var delta = eye(n).scale(h);
  var logLikelihood = this.logLikelihood(theta);
  var grad = zeros(n);
  for (var i = 0; i < n; ++i) {
    grad[i] = (this.logLikelihood(theta.add(delta.col(i))) - logLikelihood) / h;
  }
  return grad.scale(-1);
};

Regressor.prototype.gradient = function(theta) {
  var Sigma = this.computeSigma(theta);
  var dSigma = [ ];
  for (var k = 0; k < this.kernel.hyperparams.length; ++k) {
    dSigma.push(zeros(this.N, this.N));
  }
  for (var i = 0; i < this.N; ++i) {
    for (var j = i; j < this.N; ++j) {
      var grad = this.kernel.grad(this.x[i], this.x[j], theta);
      for (var k = 0; k < this.kernel.hyperparams.length; ++k) {
        dSigma[k][i * this.N + j] = grad[k];
        dSigma[k][j * this.N + i] = dSigma[k][i * this.N + j];
      }
    }
  }
  var grad = zeros(this.kernel.hyperparams.length);
  var L = Sigma.chol_inplace();
  for (var k = 0; k < this.kernel.hyperparams.length; ++k) {
    var prod = zeros(this.N, this.N);
    for (var j = 0; j < this.N; ++j) {
      prod.setCol(j, L.bsolve(L.fsolve(dSigma[k].col(j)), {transpose: true}));
    }
    grad[k] = 0.5 * prod.trace() + 0.5 * this.y.dot(dSigma[k].multiply(L.bsolve(L.fsolve(dSigma[k].multiply(this.y)), {transpose: true})));
  }
  return grad;
};

Regressor.prototype.optimize = function() {
  var theta = this.getHyperparams();
  var self = this;
  var proposal = BFGS(function(theta) { return self.gradientFD(theta); }, theta, 100);
  this.setHyperparams(proposal);
  return proposal;
};
