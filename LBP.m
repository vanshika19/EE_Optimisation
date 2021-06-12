close all;
clear all;


%%Simulation parameters

Kt = 2; %Number of base stations (BSs)
Kr = 2; %Number of users (in total)
Nt = 2; %Number of antennas per BS

rng('shuffle'); %Initiate the random number generators with a random seed

%%If rng('shuffle'); is not supported by your Matlab version, you can use
%%the following commands instead:
%randn('state',sum(100*clock));
%rand('state',sum(100*clock));

PdB = 10; %SNR (in dB)
P = 10.^(PdB/10); %SNR

%D-matrices for global joint transmission
D = repmat(eye(Kt*Nt),[1 1 Kr]);

%Definition of per base station power constraints
L = Kt;
Q = zeros(Kt*Nt,Kt*Nt,L); %The L weighting matrices
Qsqrt = zeros(Kt*Nt,Kt*Nt,L); %The matrix-square root of the L weighting matrices
for l = 1:L
    Q((l-1)*Nt+1:l*Nt,(l-1)*Nt+1:l*Nt,l) = (1/P)*eye(Nt);
    Qsqrt((l-1)*Nt+1:l*Nt,(l-1)*Nt+1:l*Nt,l) = sqrt(1/P)*eye(Nt);
end
q = ones(L,1); %Limits of the L power constraints. Note that these have been normalized.

%Generation of normalized Rayleigh fading channel realizations with unit
%variance to closest base station and 1/3 as variance from other base station.
pathlossOverNoise = 1/2*eye(Kr)+1/2*ones(Kr,Kr);
H = sqrt(kron(pathlossOverNoise,ones(1,Nt))).*(randn(Kr,Nt*Kt)+1i*randn(Kr,Nt*Kt))/sqrt(2);



%%Part 1: Calculate sample points on the Pareto boundary using Approach 1
%%in Section 3.3. The samples are computed by applying Theorem 3.4 on fine
%%grid of weighting vectors.

deltaFPO = 0.1; %Accuracy of the line searches in the FPO algorithm

nbrOfSamplesFPO = 101; %Number of sample points
ParetoBoundary = zeros(Kr,nbrOfSamplesFPO); %Pre-allocation of sample matrix


%Computation of the utopia point using MRT with full transmit power,
%which is optimal when there is only one active user.

wMRT = functionMRT(H,D); %Compute normalized beamforming vectors for MRT

utopia = zeros(Kr,1);
for k = 1:Kr
    W = sqrt(P)*[wMRT(1:2,k)/norm(wMRT(1:2,k)); wMRT(3:4,k)/norm(wMRT(3:4,k))]; %Scale to use all available power
    utopia(k) = log2(1+abs(H(k,:)*W)^2);
end


%Generate a grid of equally spaced search directions
interval = linspace(0,1,nbrOfSamplesFPO);
profileDirections = [interval; 1-interval];

for m = 1:nbrOfSamplesFPO
    
    %Output of the simulation progress - since this part takes a long time
    if mod(m,10) == 1
        disp(['Progress: ' num2str(m) ' out of ' num2str(nbrOfSamplesFPO) ' search directions']); 
    end
    
    %Search on a line from the origin in each of the equally spaced search
    %directions. The upper point is either on Pareto boundary (unlikely) or
    %outside of the rate region.
    lowerPoint = zeros(Kr,1);
    upperPoint = sum(utopia) * profileDirections(:,m);
    
    %Find the intersection between the line and the Pareto boundary by
    %solving an FPO problem.
    finalInterval = functionFairnessProfile_cvx(H,D,Qsqrt,q,deltaFPO,lowerPoint,upperPoint);
    
    ParetoBoundary(:,m) = finalInterval(:,1); %Save the best feasible point found by the algorithm
end

%Search for the sample points that maximize the sum rate. Note that the
%accuracy is improved when the number of sample points is increased.
[~,sumrateIndex] = max(ParetoBoundary(1,:)+ParetoBoundary(2,:));



%%Part 2: Run the PA and BRB algorithms to search for the point that
%%maximizes the sum rate.

disp('Progress: Running PA and BRB algorithms'); %Output of the simulation progress

%Define optimization problem as maximizing the unweighted sum rate.
problemMode = 1;
weights = ones(Kr,1);

saveBoxes = 1; %Tell PA and BRB algorithms to return the set of vertices/boxes, to enable plotting.

epsilon = 0.001; %Accuracy of the optimal sum rate in the BRB algorithm
deltaBRB = 0.001; %Accuracy of the line searches in the BRB algorithm
deltaPA=0.001; %Accuracy of the line searches in the PA algorithm

%Since the purpose of this figure is to illustrate the algorithms, we will
%not run the algorithms to convergence but only 40 iterations.
maxIterations = 30; %Maximal number of iterations of the algorithm
maxFuncEvaluations = 500; %Maximal number of convex problems solved in the algorithm

%Define the origin, which is used as the best feasible point known in
%advance (in both algorithms) and as the lower corner of the initial box in
%the BRB algorithm.
origin = zeros(Kr,1);

%Run the PA algorithm

%Run the BRB algorithm
[bestFeasibleBRB,~,~,~,boxes] = functionBRBalgorithm_cvx(H,D,Qsqrt,q,origin,utopia,weights,deltaBRB,epsilon,maxIterations,maxFuncEvaluations,origin,problemMode,saveBoxes);

disp('Progress: Finalizing'); %Output of the simulation progress



%%Plot the progress of PA and BRB algorithms.

whichIterations=[0 1 2 10 20 30]; %Which 6 iterations will be shown.



%BRB algorithm
figure(1);

for m = 1:6
        %Plot the rate region and point for maximal sum rate.
    subplot(3,2,m); hold on; box on;
    plot(ParetoBoundary(1,:),ParetoBoundary(2,:),'k');
    plot(ParetoBoundary(1,sumrateIndex),ParetoBoundary(2,sumrateIndex),'k*');
    
    if whichIterations(m) == 0
        %First subplot shows the initial box
        plot([origin(1) origin(1) utopia(1) utopia(1) origin(1)],[origin(2) utopia(2) utopia(2) origin(2) origin(2)],'b','LineWidth',0.75);
        
        legend('Pareto Boundary','Maximal Sum Utility','Location','SouthWest');
        xlabel('Initialization (BRB algorithm)');
    else
        %m:th subplot shows the set of boxes at iteration whichIterations(m).
        boxesLowerCorners = boxes{whichIterations(m)}.lowerCorners;
        boxesUpperCorners = boxes{whichIterations(m)}.upperCorners;
        
        for boxInd = 1:size(boxesUpperCorners,2)
            plot([boxesLowerCorners(1,boxInd) boxesLowerCorners(1,boxInd) boxesUpperCorners(1,boxInd) boxesUpperCorners(1,boxInd) boxesLowerCorners(1,boxInd)],[boxesLowerCorners(2,boxInd) boxesUpperCorners(2,boxInd) boxesUpperCorners(2,boxInd) boxesLowerCorners(2,boxInd) boxesLowerCorners(2,boxInd)],'b','LineWidth',0.75);
        end
        
        xlabel(['Iteration ' num2str(whichIterations(m))]);
    end
end

function [bestFeasible,Woptimal,totalNbrOfEvaluations,bounds,boxes] = functionBRBalgorithm_cvx(H,D,Qsqrt,q,boxesLowerCorners,boxesUpperCorners,weights,delta,epsilon,maxIterations,maxFuncEvaluations,bestFeasible,problemMode,saveBoxes)

%INPUT:
%H             = Kr x Kt*Nt matrix with row index for receiver and column
%                index transmit antennas
%D             = Kt*Nt x Kt*Nt x Kr diagonal matrix. Element (j,j,k) is one 
%                if j:th antenna can transmit to user k and zero otherwise
%Qsqrt         = N x N x L matrix with matrix-square roots of the L 
%              weighting matrices for the L power constraints
%q             = Limits of the L power constraints
%boxesLowerCorners = Kr x 1 vector with lower corner of an initial box that
%                    covers the rate region
%boxesUpperCorners = Kr x 1 vector with upper corner of an initial box that
%                    covers the rate region
%weights       = Kr x 1 vector with positive weights for each user
%delta         = Accuracy of the line-search in FPO subproblems 
%                (see functionFairnessProfile() for details
%epsilon       = Accuracy of the final value of the utility
%maxIterations = Maximal number of outer iterations of the algorithm
%maxFuncEvaluations = Maximal number of convex feasibility subproblems to
%                     be solved
%bestFeasible  = (Optional) Kr x 1 vector with any feasible solution
%problemMode   = (Optional) Weighted sum rate is given by mode==1 (default)
%                 Weighted proportional fairness is given by mode==2
%saveBoxes     = (Optional) Saves and return the set of boxes from each
%                 iteration of the algorithm if saveBoxes==1
%
%OUTPUT:
%bestFeasible          = The best feasible solution found by the algorithm
%Woptimal              = Kt*Nt x Kr matrix with beamforming that achieves bestFeasible
%totalNbrOfEvaluations = Vector with number of times that the convex 
%                        subproblem was solved at each iteration of the
%                        algorithm
%bounds                = Matrix where first/second column gives the global 
%                        lower/upper bound at each iteration of the algorithm
%boxes                 = Cell array where boxes{k}.lowerCorners and
%                        boxes{k}.upperCorners contain the corners of the
%                        boxes at the end of iteration k.



Kr = size(H,1);  %Number of users (in total)
I = eye(Kr); %Kr x Kr identity matrix


%Initialize the best feasible solution in the initial box as the origin or
%a point given by input
if nargin < 12
    bestFeasible = zeros(Kr,1);
    localFeasible = zeros(Kr,1);
else
    localFeasible = bestFeasible;
end


%If problemMode has not been specified: Select weighted sum rate
if nargin < 13
    problemMode = 1;
end

%If saveBoxes has not been specified: Do not save and return set of boxes
if nargin < 14
    saveBoxes = 0;
end
boxes{1}.lowerCorners=zeros(Kr,1);
boxes{1}.upperCorners=zeros(Kr,1);


%Pre-allocation of matrices for storing lower/upper bounds on optimal
%utility and the number of times the convex subproblem (power minimization 
%under QoS requirements) is solved.
lowerBound = zeros(maxIterations,1);
upperBound = zeros(maxIterations,1);
totalNbrOfEvaluations = zeros(maxIterations,1);


%Initialize current best value (cbv) and the current upper bounds (cub),
%where the latter is the potential system utility in each vertex.
if problemMode == 1 %Weighted sum rate
    cbv = weights'*localFeasible;
    cub = weights'*boxesUpperCorners;
elseif problemMode == 2 %Weighted proportional fairness
    cbv = geomean_weighted(localFeasible,weights);
    cub = geomean_weighted(boxesUpperCorners,weights);
end


%Initialize matrix for storing optimal beamforming
Woptimal = zeros(size(H'));



%Iteration of the BRB algorithm. Continue until termination by solution
%accuracy, maximum number of iterations or subproblem solutions 
for k = 1:maxIterations
    
    
    
    %Step 1 of BRB algorithm: Branch
    [~,ind] = max(cub); %Select box with current global upper bound
    
    [len,dim] = max(boxesUpperCorners(:,ind)-boxesLowerCorners(:,ind)); %Find the longest side
    
    %Divide the box into two disjoint subboxes
    newBoxesLowerCorners = [boxesLowerCorners(:,ind) boxesLowerCorners(:,ind)+I(:,dim)*len/2];
    newBoxesUpperCorners = [boxesUpperCorners(:,ind)-I(:,dim)*len/2 boxesUpperCorners(:,ind)];
    
    %Set local lower and upper bounds using Lemma 2.9
    if min(localFeasible(:,ind)>=newBoxesLowerCorners(:,2)) == 1
        point = localFeasible(:,ind)-newBoxesUpperCorners(:,1);
        point(point<0) = 0;
        localFeasibleNew = [localFeasible(:,ind)-point localFeasible(:,ind)];
    else
        localFeasibleNew = [localFeasible(:,ind) localFeasible(:,ind)];
    end
    
    
    %Step 2 of BRB algorithm: Reduce
    
    %Reduction if the two new boxes based on Lemma 2.10
    if problemMode == 1
        
        %Reduction for weighted sum rate is given by Example 2.11
        cubNew = min([weights'*newBoxesUpperCorners; cub(ind)*ones(1,2)],[],1);
        
        newBoxesLowerCornersReduced = zeros(size(newBoxesLowerCorners));
        for m = 1:Kr
            nu = (weights'*newBoxesUpperCorners-cbv)./(weights(m)*(newBoxesUpperCorners(m,:)-newBoxesLowerCorners(m,:)));
            nu(nu>1) = 1;
            newBoxesLowerCornersReduced(m,:) = (1-nu).*newBoxesUpperCorners(m,:)+nu.*newBoxesLowerCorners(m,:);
        end
        
        newBoxesUpperCornersReduced = zeros(size(newBoxesUpperCorners));
        for m = 1:Kr
            mu = (cubNew-weights'*newBoxesLowerCornersReduced)./(weights(m)*(newBoxesUpperCorners(m,:)-newBoxesLowerCornersReduced(m,:)));
            mu(mu>1) = 1;
            newBoxesUpperCornersReduced(m,:) = (1-mu).*newBoxesLowerCornersReduced(m,:)+mu.*newBoxesUpperCorners(m,:);
        end
        
    elseif problemMode == 2
        
        %Reduction for weighted proportional fairness is given by Example 2.12
        cubNew = min([geomean_weighted(newBoxesUpperCorners,weights); cub(ind)*ones(1,2)],[],1);
        
        newBoxesLowerCornersReduced = zeros(size(newBoxesLowerCorners));
        for m = 1:Kr
            nu = (1-(cbv^(1/weights(m)))./(prod(newBoxesUpperCorners.^(repmat(weights,[1 2])/weights(m)),1))) .* newBoxesUpperCorners(m,:) ./(newBoxesUpperCorners(m,:)-newBoxesLowerCorners(m,:));
            nu(nu>1) = 1;
            nu(isnan(nu)) = 1;
            newBoxesLowerCornersReduced(m,:) = (1-nu).*newBoxesUpperCorners(m,:)+nu.*newBoxesLowerCorners(m,:);
        end
        
        newBoxesUpperCornersReduced = zeros(size(newBoxesUpperCorners));
        for m = 1:Kr
            mu = ((cubNew.^(1/weights(m)))./prod(newBoxesLowerCornersReduced.^(repmat(weights,[1 2])/weights(m)),1)-1) .* newBoxesLowerCornersReduced(m,:) ./ (newBoxesUpperCorners(m,:)-newBoxesLowerCorners(m,:));
            mu(mu>1) = 1;
            mu(isnan(mu)) = 1;
            newBoxesUpperCornersReduced(m,:) = (1-mu).*newBoxesLowerCornersReduced(m,:)+mu.*newBoxesUpperCorners(m,:);
        end
        
    end
    
    %Update the two new boxes with the reduced versions
    newBoxesLowerCorners = newBoxesLowerCornersReduced;
    newBoxesUpperCorners = newBoxesUpperCornersReduced;
    
    
    %Step 3 of BRB algorithm: Bound
    
    %Check if lower corners of the two boxes are feasible
    feasible = zeros(2,1); %Set l:th element to one if the l:th box is feasible
    totalNbrOfEvaluations(k) = 0; %Number of convex problems solved in k:th iteration 
    for l = 1:2
        
        %Compute potential performance in upper corner of l:th new box
        if problemMode == 1
            localUpperBoundl = weights'*newBoxesUpperCorners(:,l);
        elseif problemMode == 2
            localUpperBoundl = geomean_weighted(newBoxesUpperCorners(:,l),weights);
        end
        
        %Check if potential performance is better than current best
        %feasible solution
        if localUpperBoundl>cbv
            
            %Check if the current local feasible point is actually in the
            %box (it might not in the current box due to branching and
            %reduction procedures)
            if min(localFeasibleNew(:,l)>=newBoxesLowerCorners(:,l))==1
                %The lower corner is feasible since there we have a local
                %feasible point that strictly dominates this point
                feasible(l) = 1; 
            else
                %The feasibility of the lower corner cannot be determined
                %by previous information and we need to solve a convex
                %feasibility problem.
                gammavar = 2.^(newBoxesLowerCorners(:,l))-1; %Transform lower corner into SINR requirements 
                [checkFeasibility,W] = functionFeasibilityProblem_cvx(H,D,Qsqrt,q,gammavar); %Solve the feasibility problem
                
                totalNbrOfEvaluations(k) = totalNbrOfEvaluations(k)+1; %Increase number of feasibility evaluations in k:th iteration
                
                %Check if the point was feasible
                if checkFeasibility == false
                    feasible(l) = 0; %Not feasible
                elseif checkFeasibility == true
                    feasible(l) = 1; %Feasible
                    localFeasibleNew(:,l) = newBoxesLowerCorners(:,l); %Update local feasible point
                    
                    %Compute the performance in the lower corner
                    if problemMode==1
                        localLowerBoundl = weights'*newBoxesLowerCorners(:,l);
                    elseif problemMode==2
                        localLowerBoundl = geomean_weighted(newBoxesLowerCorners(:,l),weights);
                    end
                    
                    %If the feasible point is better than all found so far,
                    %then it is stored.
                    if localLowerBoundl>cbv
                        bestFeasible = newBoxesLowerCorners(:,l);
                        cbv = localLowerBoundl;
                        Woptimal = W;
                    end
                end
            end
            
        else
            feasible(l) = 0; %The box whole box is inside the rate region and can be removed
        end

    end
    

    %Search for a better feasible point in the outmost of the new boxes.
    %The search is only done if the box is feasible.
    if feasible(2)==1
        
        %Solve an FPO problem to find a better feasible point the outmost
        %box. We search for a point on the line between the lower and upper
        %corner, with an accuracy given by delta.
        [interval,W,FPOevaluations] = functionFairnessProfile_cvx(H,D,Qsqrt,q,delta,newBoxesLowerCorners(:,2),newBoxesUpperCorners(:,2));
        
        %Update the number of feasibility evaluations
        totalNbrOfEvaluations(k) = totalNbrOfEvaluations(k)+FPOevaluations; 
        
        %The new feasible point found by the FPO problem
        newFeasiblePoint = interval(:,1);
        
        %The point interval(:,2) is either infeasible or on the Pareto
        %boundary. All points in the box that strictly dominate this point
        %can be ignored. The Kr corner points of the remaining polyblock 
        %are calculated and will be used to improve the local upper bound.
        reduced_corners = repmat(newBoxesUpperCorners(:,2),[1 Kr])-diag(newBoxesUpperCorners(:,2)-interval(:,2));
        
        %A local upper bound can be computed as the largest system utility
        %achieved among the Kr corner points computed above. This new bound 
        %replaces the current local upper bound if it is smaller.
        if problemMode == 1
            cubNew(2) = min([max(weights'*reduced_corners) cubNew(2)]);
        elseif problemMode == 2
            cubNew(2) = min([max(geomean_weighted(reduced_corners,weights)) cubNew(2)]);
        end
        
        %Update the local feasible point, if the new point is better.
        %Update the global feasible point, if the new point is better.
        if problemMode == 1
            
            if weights'*newFeasiblePoint > weights'*localFeasibleNew(:,2)
                localFeasibleNew(:,2) = newFeasiblePoint;
            end
            
            if weights'*newFeasiblePoint > cbv
                bestFeasible = newFeasiblePoint;
                cbv = weights'*newFeasiblePoint;
                Woptimal = W; %Store beamforming for current best solution
            end
            
        elseif problemMode == 2
            
            if geomean_weighted(newFeasiblePoint,weights) > geomean_weighted(localFeasibleNew(:,2),weights)
                localFeasibleNew(:,2) = newFeasiblePoint;
            end
            
            if geomean_weighted(newFeasiblePoint,weights) > cbv
                bestFeasible = newFeasiblePoint;
                cbv = geomean_weighted(newFeasiblePoint,weights);
                Woptimal = W; %Store beamforming for current best solution
            end
            
        end
        
    end
    
    
    %Step 4 of BRB algorithm: Prepare for next iteration
    
    %Check which boxes that should be kept for the next iteration
    keep = cub>cbv; %Only keep boxes that might contain better points than current best solution
    keep(ind) = false; %Remove the box that was branched
    keepnew = (feasible==1); %Only keep new boxes that are feasible
    
    %Update the boxes and their local information for the next iteration
    %of the algorithm.
    boxesLowerCorners = [boxesLowerCorners(:,keep) newBoxesLowerCorners(:,keepnew)];
    boxesUpperCorners = [boxesUpperCorners(:,keep) newBoxesUpperCorners(:,keepnew)];
    cub = [cub(keep) cubNew(keepnew)];
    localFeasible = [localFeasible(:,keep) localFeasibleNew(:,keepnew)];
    
    %Store the lower and upper bounds in the k:th iteration to enable
    %plotting of the progress of the algorithm.
    lowerBound(k) = cbv;
    upperBound(k) = max(cub);
    
    if saveBoxes == 1
        boxes{k}.lowerCorners=boxesLowerCorners;
        boxes{k}.upperCorners=boxesUpperCorners;
    end
    
    %Check termination conditions
    if sum(totalNbrOfEvaluations) >= maxFuncEvaluations %Maximal number of feasibility evaluations has been used
        break;
    elseif upperBound(k)-lowerBound(k) <= epsilon %Predefined accuracy of optimal solution has been achieved
        break;
    end
end


%Prepare output by removing parts of output vectors that were not used
totalNbrOfEvaluations = totalNbrOfEvaluations(1:k);
bounds = [lowerBound(1:k) upperBound(1:k)];

end


function y = geomean_weighted(x,w)
%Calculate weighted proportional fairness of each column of z.
%The corresponding weights are given in w

y = prod(x.^repmat(w,[1 size(x,2)]),1);
end

function [finalInterval,WBestBeamforming,nbrOfEvaluations] = functionFairnessProfile_cvx(H,D,Qsqrt,q,delta,lowerPoint,upperPoint,specialMode,specialParam)


%INPUT:
%H           = Kr x Kt*Nt matrix with row index for receiver and column
%              index transmit antennas
%D           = Kt*Nt x Kt*Nt x Kr diagonal matrix. Element (j,j,k) is one if
%              j:th transmit antenna can transmit to user k and zero otherwise
%Qsqrt       = N x N x L matrix with matrix-square roots of the L weighting
%              matrices for the L power constraints
%q           = Limits of the L power constraints
%delta       = Accuracy of the final solution. The algorithm terminates when
%                 norm(upperPoint - lowerPoint) <= delta
%lowerPoint  = Start point of the line (must be inside the rate region)
%upperPoint  = End point of the line (must be outside of the rate region)
%specialMode = (Optional) Consider different non-idealities in the system
%              model. Normally we have specialMode==0. 
%              Transceiver impairments is given by specialMode==1.
%specialParam = (Optional) Additional parameters for each specialMode.
%               specialMode==1: 2 x 1 vector with EVM at the transmit and 
%               receive antennas, respectively. These are used along with
%               linear distortion functions.
%
%OUTPUT:
%finalInterval    = Kr x 2 matrix with lowerPoint and upperPoint at
%                   termination
%WBestBeamforming = Kt*Nt x Kr matrix with beamforming that achieves the 
%                   lower point in the final interval
%nbrOfEvaluations = Number of times that the convex subproblem (power
%                   minimization under QoS requirements) is solved

if nargin < 8
    specialMode = 0;
end

Kr = size(H,1); %Number of users
L = size(Qsqrt,3); %Number of power constraints


%Pre-allocation of matrix for storing optimal beamforming
WBestBeamforming = [];

%Count the number of feasibility problem solved
nbrOfEvaluations = 0;

%%Part 1: Solve the problem by bisection.

%Solve the problem by bisection - iterate until different between
%current lower and upper point
while norm(upperPoint - lowerPoint) > delta
    
    candidatePoint = (lowerPoint+upperPoint)/2; %Compute the midpoint at the line
    
    gammavar = 2.^(candidatePoint)-1; %Transform midpoint into SINR requirements
    
    %Check the feasibility at the midpoint by solving a feasibility
    %problem. Different feasibility problems are solved depending on the
    %mode. 
    if specialMode == 0 %Ideal case
        [feasible,Wcandidate] = functionFeasibilityProblem_cvx(H,D,Qsqrt,q,gammavar);
    elseif specialMode == 1 %Transceiver impairments
        [feasible,Wcandidate] = functionFeasibilityProblem_Impairments_cvx(H,D,Qsqrt,q,gammavar,specialParam);
    end
    
    %If the problem was feasible, then replace lowerPoint with
    %candidatePoint and store W as current best solution.
    if feasible
        lowerPoint = candidatePoint;
        WBestBeamforming = Wcandidate;
    else
        %If the problem was not feasible,then replace upperPoint with candidatePoint
        upperPoint = candidatePoint;
    end
    
    %Increase the number of function evaluations
    nbrOfEvaluations = nbrOfEvaluations+1;
end




%%Part 2: Prepare the achieved solution for output

%If the midpoints analyzed by the algorithm have never been feasible,
%then obtain a feasible beamforming solution using the lowerPoint. This
%happens when delta is too large or when the optimal point is very
%close to lowerPoint.
if isempty(WBestBeamforming)
    gammavar = 2.^(lowerPoint)-1;
    
    if specialMode == 0 %Ideal case
        [feasible,Wcandidate] = functionFeasibilityProblem_cvx(H,D,Qsqrt,q,gammavar);
    elseif specialMode == 1 %Transceiver impairments
        [feasible,Wcandidate] = functionFeasibilityProblem_Impairments_cvx(H,D,Qsqrt,q,gammavar,specialParam);
    end
    
    if feasible
        WBestBeamforming = Wcandidate;
    else
        %The algorithm requires that the start point is inside of the
        %rate region, which is not the case if we end up here.
        error('Fairness-profile optimization problem is infeasible');
    end
end

%Prepare for output, depending on scenario mode
if specialMode == 0 %Ideal case
    
    %Change scaling of achieved beamforming to satisfy at least one power
    %constraint with equality (based on Theorem 1.2).
    scaling = zeros(L,1);
    for l = 1:L
        scaling(l) = norm(Qsqrt(:,:,l)*WBestBeamforming,'fro').^2/q(l);
    end
    
    %Scale beamforming to satisfy at least one power constraint with equality
    WBestBeamforming = WBestBeamforming/sqrt(max(scaling));
    
    
    %Compute the rates that are actually achieved by WBestBeamforming
    channelGains = abs(H*WBestBeamforming).^2;
    signalGains = diag(channelGains);
    interferenceGains = sum(channelGains,2)-signalGains;
    rates = log2(1+signalGains./(1+interferenceGains));
    
    %Store the final interval between lower and upper point
    if sum(rates>lowerPoint) == Kr
        finalInterval = [rates upperPoint];
    else
        finalInterval = [lowerPoint upperPoint];
    end
    
elseif specialMode == 1 %Transceiver impairments
    
    %Store the final interval between lower and upper point
    finalInterval = [lowerPoint upperPoint];
    
end
end

function wMRT = functionMRT(H,D)

%INPUT:
%H  = Kr x Kt*Nt matrix with row index for users and column index
%     transmit antennas
%D  = Kt*Nt x Kt*Nt x Kr diagonal matrix. Element (j,j,k) is one if j:th
%     transmit antenna can transmit to user k and zero otherwise
%
%OUTPUT:
%wMRT = Kt*Nt x Kr matrix with normalized MRT beamforming



%Number of users
Kr = size(H,1);

%Total number of antennas
N = size(H,2);

%If D matrix is not provided, all antennas can transmit to everyone
if nargin<2
    D = repmat( eye(N), [1 1 Kr]);
end

%Pre-allocation of MRT beamforming
wMRT = zeros(size(H'));

%Computation of MRT, based on Definition 3.2
for k = 1:Kr
    channelvector = (H(k,:)*D(:,:,k))'; %Useful channel
    wMRT(:,k) = channelvector/norm(channelvector); %Normalization of useful channel
end
end
function [feasible,Wsolution] = functionFeasibilityProblem_cvx(H,D,Qsqrt,q,gammavar)

%INPUT:
%H          = Kr x Kt*Nt matrix with row index for receiver and column
%             index transmit antennas
%D          = Kt*Nt x Kt*Nt x Kr diagonal matrix. Element (j,j,k) is one if
%             j:th transmit antenna can transmit to user k and zero otherwise
%Qsqrt      = N x N x L matrix with matrix-square roots of the L weighting 
%             matrices for the L power constraints
%q          = Limits of the L power constraints
%gammavar   = Kr x 1 vector with SINR constraints for all users.
%
%OUTPUT:
%feasible  = This variable is feasible=true if the feasibility problem is
%            feasible. Otherwise we have feasible=false.
%Wsolution = Kt*Nt x Kr matrix with beamforming achieved by the power
%            minimization problem. This matrix is empty if this problem is
%            infeasible.


Kr = size(H,1); %Number of users
N = size(H,2); %Number of transmit antennas (in total)
L = size(Qsqrt,3); %Number of power constraints


%Solve the power minimization under QoS requirements problem using CVX
cvx_begin
cvx_quiet(true); % This suppresses screen output from the solver

variable W(N,Kr) complex;  %Variable for N x Kr beamforming matrix
variable betavar %Scaling parameter for power constraints

minimize betavar %Minimize the power indirectly by scaling power constraints

subject to

%SINR constraints (Kr constraints)
for k = 1:Kr
    
    %Channels of the signal intended for user i when it reaches user k
    hkD = zeros(Kr,N);
    for i = 1:Kr
        hkD(i,:) = H(k,:)*D(:,:,i);
    end
    
    imag(hkD(k,:)*W(:,k)) == 0; %Useful link is assumed to be real-valued
    
    %SOCP formulation for the SINR constraint of user k
    real(hkD(k,:)*W(:,k)) >= sqrt(gammavar(k))*norm([1 hkD(k,:)*W(:,[1:k-1 k+1:Kr])  ]);
end

%Power constraints (L constraints) scaled by the variable betavar
for l = 1:L
    norm(Qsqrt(:,:,l)*W,'fro') <= betavar*sqrt(q(l));
end

betavar >= 0; %Power constraints must be positive

cvx_end


%Analyze result and prepare the output variables.
if isempty(strfind(cvx_status,'Solved')) %Both power minimization problem and feasibility problem are infeasible.
    feasible = false;
    Wsolution = [];
elseif betavar>1 %Only power minimization problem is feasible.
    feasible = false;
    Wsolution = W;
else %Both power minimization problem and feasibility problem are feasible.
    feasible = true;
    Wsolution = W;
end
end

