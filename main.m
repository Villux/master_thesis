
%% fixed parameters
T = 10; % Two years of data
dt = (1/(60*6.5))/252; % Frequency one day
n = T/dt;
disp([num2str(n), ' observations in each sample']);

parameters.gamma = [0.5];

M = 50000; % Number of simulated samples for the combination of parameters

%% Parameter sets
kappa = [0.2, 2, 6];
theta = [0.1^2, 0.3^2, 0.5^2];
rho = [-0.1, -0.5, -0.9];
xi = [0.1, 0.3, 0.6];

%%
for i_kappa = 1:length(kappa)
    for i_theta = 1:length(theta)
        for i_rho = 1:length(rho)
            for i_xi = 1:length(theta)
                for i_M = 1:M
                    seedNumber = datenum(clock); % seed not fixed
                    
                    parameters.kappa = kappa(i_kappa);
                    parameters.theta = theta(i_theta);
                    parameters.rho = rho(i_rho);
                    parameters.xi = xi(i_xi);
                    parameters.v0 = parameters.theta;
                    
                    [returnData{i_kappa, i_theta, i_rho, i_xi, i_M}, ...
                        volatilityData{i_kappa, i_theta, i_rho, i_xi, i_M}] = ...
                        dataGeneration(parameters, seedNumber, T, dt);
                end
            end
        end
    end
end