

%fpdf = @(x) double(abs(x) <= 1.5) .* normpdf( x );
fpdf = @(x)((1/sqrt(2*pi))*exp(-(x-3)^2 / 2));
a=0.5;
b=2;
%fpdf = @(x)((b/a)*(x/a)^(b-1)*exp(-(x/a)^b));
guess = 0;
num_iterations = 5000;
random_draws = MetropolisHastings(guess, num_iterations, fpdf);
use_draws = random_draws(50:end); % recall that the algorithm has a burn-in
figure(2)
hist(random_draws)