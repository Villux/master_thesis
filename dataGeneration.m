function [returnData, volatilityData] = ...
    dataGeneration(parameters, seedNumber, T, dt)

%%
rng(seedNumber,'v5normal')

n = T/dt; % Number of steps

%% Parameters
kappa = parameters.kappa;
theta = parameters.theta;
xi = parameters.xi;
rho = parameters.rho;
gamma = parameters.gamma;
v0 = parameters.v0;


%% Simulation
v = zeros(n, 1); v(1) = v0;
s = zeros(n, 1); s(1) = 1;

epsilon_s = randn(n, 1);
epsilon_v = rho.*epsilon_s+sqrt(1-rho^2)*randn(n, 1);

for i = 2:n
    s(i) = s(i-1).*exp(-0.5*v(i-1)*dt + sqrt(v(i-1)*dt)*epsilon_s(i-1));
    v(i) = max(v(i-1) + kappa*(theta - v(i-1))*dt + xi*v(i-1)^gamma*epsilon_v(i-1)*sqrt(dt), 0);
end

returnData = log(s(2:end)./s(1:end-1));
volatilityData = v(1:end-1);
%volatilityData = v(2:end) - v(1:end-1);

end