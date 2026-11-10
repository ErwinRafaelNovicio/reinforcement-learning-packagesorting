classdef RL_environment < rl.env.MATLABEnvironment
    %AMS_RL: Template for defining custom environment in MATLAB.    
    
    %% Properties (set properties' attributes accordingly)
    properties
        % Specify and initialize environment's necessary properties
        % Velocity of the exit conveyor
        v_treadmill = 1.9; % m/s, also change to 1.6 to speed up the training
        % Size of each AMS tile 0.2 x 0.2 m
        d_AMS = 0.2; % m - size of the AMS
        % AMS Matrix Grid Dimensions (5x5)
        n_i_AMS = 5; % numeber of AMS along y
        n_j_AMS = 5; % number of AMS along x
        % Number of actions of each AMS: rotation and velocity
        n_actions_art = 2; % number of actions of each AMS
        
        % Rotation and Velocity limits
        upperlim_rot = pi*45/180; % upper limit rotation for AMS action [deg]
        upperlim_v = 2.2; % max velocity for AMS [m/s]
        lowerlim_rot = - pi*45/180; % lower limit rotation for AMS action [deg]
        lowerlim_v = 0.5; % min velocity for AMS [m/s]
        
        % Objects to store Parcel States
        vy_boxes = []; % Stores y-direction velocity of Parcels
        % Store previous positions -> used for collision detection and motion analysis
        x_boxes_prec = [];
        y_boxes_prec = [];
        
        % Stores which AMS each parcel is currently on
        index_AMS = [];
        
        % Stores previous control actions
        LastAction = zeros(50,1);
        
        % Stores the AMS row/column under each parcel.
        i_box = [];
        j_box = [];
        
        % Tracks how many parcels were added at each generation event.
        n_boxes_tot_vect = [];
        % Current number of parcels in system.
        n_boxes_tot = 0;
        % Max number of parcels: episode contains at most 10 parcels.
        max_gen_boxes = 10;
        % Generation counter: counts parcel generation events.
        cont_new_box = 0;
        
        % Parcel geometry and trajectories
        % Diameter/size of each parcel.
        d_boxes = zeros(10,1);
        % Current parcel coordinates.
        x_boxes = zeros(10,1);
        y_boxes = zeros(10,1);
        
        % Full trajectory history FIXED to: [timesteps x parcels]
        x_boxes_vect = zeros(10000,10);
        y_boxes_vect = zeros(10000,10);
        
        % Timestep counter: Counts simulation steps.
        cont = 0;
        % Exit tracking: Has parcel exited?
        pack_exited = zeros(10,1);
        % Exit order
        exit_order = zeros(10,1);
        % Exit counter
        index_exit = 0;
        
        % Collision tollerance
        toll_contatto = 0.005;
        dt = 0.01; % s - timestep for simulation
        
        % Simulation clock
        time = 0; % s - simulation time
    end
    
    properties (Hidden)
        VisualizeAnimation = true
        VisualizeActions = false
        VisualizeStates = false        
    end
    
    properties
        State = zeros(100,1)
    end
    
    properties(Access = protected)
        IsDone = false        
    end
    properties (Transient, Access = private)
        Visualizer = []
    end
    
    methods              
        function this = RL_environment()
            nObs = 4*25; 
            ObservationInfo = rlNumericSpec([nObs 1]); 
            ObservationInfo.Name = 'System States';
            ObservationInfo.Description = 'Flattened 100x1 vector: [x, y, size, flag] for each of the 25 cells';
            
            n_actions = 5*5*2;
            ActionInfo = rlNumericSpec([n_actions 1]);
            ActionInfo.Name = 'AMS Action';
            ActionInfo.Description = 'r1, v1, r2, v2, ...';
            ActionInfo.LowerLimit = zeros(n_actions,1);
            ActionInfo.UpperLimit = zeros(n_actions,1);
            for ii=1:2:n_actions
                ActionInfo.UpperLimit(ii) = pi*45/180;
                ActionInfo.UpperLimit(ii+1) = 2.2;
                ActionInfo.LowerLimit(ii) = - pi*45/180;
                ActionInfo.LowerLimit(ii+1) = 0.5;
            end
            this = this@rl.env.MATLABEnvironment(ObservationInfo,ActionInfo);
        end
        
        function [Observation,Reward,IsDone,Info] = step(this,Action)
            Info = [];
            this.time = this.time + this.dt;
            this.cont = this.cont + 1;
            l_AMS_matrix = this.d_AMS*this.n_j_AMS; 
            
            ActLimUp = zeros(50,1);
            ActLimLow = zeros(50,1);
            for ii=1:2:50
                ActLimUp(ii) = this.upperlim_rot;
                ActLimUp(ii+1) = this.upperlim_v;
                ActLimLow(ii) = this.lowerlim_rot;
                ActLimLow(ii+1) = this.lowerlim_v;
            end
            
            AMS_actions = ActLimLow + (1 + Action) .* (ActLimUp - ActLimLow)./2;
            for ii=1:50
                AMS_actions(ii) = max(ActLimLow(ii),min(ActLimUp(ii),AMS_actions(ii)));
            end
            this.LastAction = AMS_actions;
            v_AMS = zeros(5,5);
            rotation_AMS = zeros(5,5);
            for ii = 1:this.n_i_AMS
                for jj = 1:this.n_j_AMS
                    rotation_AMS(ii,jj) = AMS_actions(jj*2-1+(ii-1)*10);
                    v_AMS(ii,jj) = AMS_actions(jj*2+(ii-1)*10);
                end
            end
            
            if mod(round(this.time,2),0.75) == 0 && this.n_boxes_tot<this.max_gen_boxes && this.time>0.25
                this.cont_new_box = this.cont_new_box + 1;
                gen_n_boxes = 2;
                while this.n_boxes_tot+gen_n_boxes>this.max_gen_boxes
                    gen_n_boxes = gen_n_boxes-1;
                end
                this.n_boxes_tot = this.n_boxes_tot+gen_n_boxes;
                this.n_boxes_tot_vect(this.cont_new_box) = gen_n_boxes;
                for ii=1:gen_n_boxes
                    if gen_n_boxes>1
                        if ii == 1
                            this.x_boxes(this.n_boxes_tot-1) = this.d_boxes(this.n_boxes_tot-1)*0.5 + (this.d_AMS*2.5-this.d_boxes(this.n_boxes_tot-1)*0.5)*rand(1);
                            this.y_boxes(this.n_boxes_tot-1) = 0.001 + rand(1)*0.01;
                            this.x_boxes_prec(this.n_boxes_tot-1) = this.x_boxes(this.n_boxes_tot-1);
                            this.y_boxes_prec(this.n_boxes_tot-1) = this.y_boxes(this.n_boxes_tot-1);
                        else
                            this.x_boxes(this.n_boxes_tot) = this.d_AMS*2.5+this.d_boxes(this.n_boxes_tot)*0.5 + (l_AMS_matrix-(this.d_AMS*2.5+this.d_boxes(this.n_boxes_tot)*0.5))*rand(1);
                            this.y_boxes(this.n_boxes_tot) = 0.001 + rand(1)*0.05;
                            if abs(this.x_boxes(this.n_boxes_tot)-this.x_boxes(this.n_boxes_tot-1)) <= this.d_boxes(this.n_boxes_tot)/2+this.d_boxes(this.n_boxes_tot-1)/2
                                this.x_boxes(this.n_boxes_tot) = this.x_boxes(this.n_boxes_tot-1) + this.d_boxes(this.n_boxes_tot)/2 + this.d_boxes(this.n_boxes_tot-1)/2 + 0.1;
                            end
                            if this.x_boxes(this.n_boxes_tot)-this.d_boxes(this.n_boxes_tot)/2<0
                                this.x_boxes(this.n_boxes_tot) = this.d_boxes(this.n_boxes_tot)/2;
                            elseif this.x_boxes(this.n_boxes_tot)+this.d_boxes(this.n_boxes_tot)/2>l_AMS_matrix
                                this.x_boxes(this.n_boxes_tot) = l_AMS_matrix-this.d_boxes(this.n_boxes_tot)/2;
                            end
                            if abs(this.x_boxes(this.n_boxes_tot)-this.x_boxes(this.n_boxes_tot-1)) <= this.d_boxes(this.n_boxes_tot)/2+this.d_boxes(this.n_boxes_tot-1)/2
                                this.x_boxes(this.n_boxes_tot-1) = this.x_boxes(this.n_boxes_tot) - (this.d_boxes(this.n_boxes_tot)/2 + this.d_boxes(this.n_boxes_tot-1)/2 + 0.1);
                            end
                            this.x_boxes_prec(this.n_boxes_tot) = this.x_boxes(this.n_boxes_tot);
                            this.y_boxes_prec(this.n_boxes_tot) = this.y_boxes(this.n_boxes_tot);
                        end
                    else
                        this.x_boxes(this.n_boxes_tot) = this.d_boxes(this.n_boxes_tot)*0.55 + (l_AMS_matrix-this.d_boxes(this.n_boxes_tot)*0.55)*rand(1);
                        this.y_boxes(this.n_boxes_tot) = 0.001 + rand(1)*0.01;
                        this.x_boxes_prec(this.n_boxes_tot) = this.x_boxes(this.n_boxes_tot);
                        this.y_boxes_prec(this.n_boxes_tot) = this.y_boxes(this.n_boxes_tot);
                        % FIXED INDEX ORDER HERE
                        this.x_boxes_vect(this.cont-1, this.n_boxes_tot) = this.x_boxes(this.n_boxes_tot);
                        this.y_boxes_vect(this.cont-1, this.n_boxes_tot) = this.y_boxes(this.n_boxes_tot);
                    end
                end
            end
            
            for ii=this.n_boxes_tot+1:this.max_gen_boxes
                this.x_boxes(ii) = -1;
                this.y_boxes(ii) = -1;
            end
            act_AMS = zeros(25,1);
            this.index_AMS = zeros(this.max_gen_boxes, 1);
            
            for ii=1:this.n_boxes_tot
                if this.x_boxes(ii)-this.d_boxes(ii)/2<0
                    this.x_boxes(ii) = this.d_boxes(ii)/2;
                elseif this.x_boxes(ii)+this.d_boxes(ii)/2>l_AMS_matrix
                    this.x_boxes(ii) = l_AMS_matrix-this.d_boxes(ii)/2;
                end
                
                % Fast logical indexing for grid positions
                this.i_box(ii) = min(6, floor(this.y_boxes(ii) / this.d_AMS) + 1);
                if this.i_box(ii) == 6
                    this.j_box(ii) = 6;
                else
                    this.j_box(ii) = min(5, floor(this.x_boxes(ii) / this.d_AMS) + 1);
                end
                
                if this.i_box(ii) < 6
                    this.x_boxes(ii) = this.x_boxes(ii) + v_AMS(this.i_box(ii),this.j_box(ii))*this.dt*sin(rotation_AMS(this.i_box(ii),this.j_box(ii)));
                    this.y_boxes(ii) = this.y_boxes(ii) + v_AMS(this.i_box(ii),this.j_box(ii))*this.dt*cos(rotation_AMS(this.i_box(ii),this.j_box(ii)));
                    this.vy_boxes(ii) = v_AMS(this.i_box(ii),this.j_box(ii))*this.dt*cos(rotation_AMS(this.i_box(ii),this.j_box(ii)));
                else
                    this.y_boxes(ii) = this.y_boxes(ii) + this.v_treadmill*this.dt;
                    this.vy_boxes(ii) = this.v_treadmill;
                end
                
                if ii>1 && this.cont>1
                    for jj=ii:-1:2
                        if abs(this.x_boxes(ii)-this.x_boxes(jj-1)) <= this.d_boxes(ii)/2+this.d_boxes(jj-1)/2 + 0.001 ...
                                && abs(this.x_boxes_vect(this.cont-1, ii)-this.x_boxes_vect(this.cont-1, jj-1)) >= this.d_boxes(ii)/2+this.d_boxes(jj-1)/2 ...
                                && abs(this.y_boxes(ii)-this.y_boxes(jj-1)) <= this.d_boxes(ii)/2+this.d_boxes(jj-1)/2 ...
                                && abs(this.y_boxes_vect(this.cont-1, ii)-this.y_boxes_vect(this.cont-1, jj-1)) <= this.d_boxes(ii)/2+this.d_boxes(jj-1)/2
                            if (this.x_boxes(ii)-this.d_boxes(ii)/2 <= this.x_boxes(jj-1)+this.d_boxes(jj-1)/2 && this.x_boxes(ii)-this.d_boxes(ii)/2>this.x_boxes(jj-1)-this.d_boxes(jj-1)/2)
                                penetrazione_x = (this.x_boxes(jj-1)+this.d_boxes(jj-1)/2) - (this.x_boxes(ii)-this.d_boxes(ii)/2);
                                this.x_boxes(ii) = this.x_boxes(ii) + penetrazione_x/2 + this.toll_contatto;
                                this.x_boxes(jj-1) = this.x_boxes(jj-1) - penetrazione_x/2 - this.toll_contatto;
                            elseif (this.x_boxes(ii)+this.d_boxes(ii)/2 >= this.x_boxes(jj-1)-this.d_boxes(jj-1)/2 && this.x_boxes(ii)+this.d_boxes(ii)/2<this.x_boxes(jj-1)+this.d_boxes(jj-1)/2)
                                penetrazione_x = (this.x_boxes(ii)+this.d_boxes(ii)/2) - (this.x_boxes(jj-1)-this.d_boxes(jj-1)/2);
                                this.x_boxes(ii) = this.x_boxes(ii) - penetrazione_x/2 - this.toll_contatto;
                                this.x_boxes(jj-1) = this.x_boxes(jj-1) + penetrazione_x/2 + this.toll_contatto;
                            end
                        elseif abs(this.y_boxes(ii)-this.y_boxes(jj-1)) <= this.d_boxes(ii)/2+this.d_boxes(jj-1)/2 + 0.001 ...
                                && abs(this.y_boxes_vect(this.cont-1, ii)-this.y_boxes_vect(this.cont-1, jj-1)) >= this.d_boxes(ii)/2+this.d_boxes(jj-1)/2 ...
                                && abs(this.x_boxes(ii)-this.x_boxes(jj-1)) <= this.d_boxes(ii)/2+this.d_boxes(jj-1)/2 
                            if (this.y_boxes(ii)-this.d_boxes(ii)/2 <= this.y_boxes(jj-1)+this.d_boxes(jj-1)/2 && this.y_boxes(ii)-this.d_boxes(ii)/2>this.y_boxes(jj-1)-this.d_boxes(jj-1)/2)
                                penetrazione_y = (this.y_boxes(jj-1)+this.d_boxes(jj-1)/2)-(this.y_boxes(ii)-this.d_boxes(ii)/2);
                                this.y_boxes(ii) = this.y_boxes(ii) + penetrazione_y/2 + this.toll_contatto;
                                this.y_boxes(jj-1) = this.y_boxes(jj-1) - penetrazione_y/2 - this.toll_contatto;
                            elseif (this.y_boxes(ii)+this.d_boxes(ii)/2 < this.y_boxes(jj-1)+this.d_boxes(jj-1)/2 && this.y_boxes(ii)+this.d_boxes(ii)/2 >= this.y_boxes(jj-1)-this.d_boxes(jj-1)/2)
                                penetrazione_y = (this.y_boxes(ii)+this.d_boxes(ii)/2) - (this.y_boxes(jj-1)-this.d_boxes(jj-1)/2);
                                this.y_boxes(ii) = this.y_boxes(ii) - penetrazione_y/2  - this.toll_contatto;
                                this.y_boxes(jj-1) = this.y_boxes(jj-1) + penetrazione_y/2 + this.toll_contatto;
                            end
                        end
                    end
                end
                
                if this.i_box(ii)<6 && this.j_box(ii)<6
                    act_AMS(this.j_box(ii)+(this.i_box(ii)-1)*5) = 1;
                    this.index_AMS(ii) = this.j_box(ii)+(this.i_box(ii)-1)*5;
                else
                    this.index_AMS(ii) = 0;
                end
            end
            
            nObs = prod(this.getObservationInfo.Dimension);
            obs = zeros(nObs, 1);
            l_max = this.d_AMS * 5; 
            
            for row = 1:5
                for col = 1:5
                    idx = (col + (row-1)*5 - 1) * 4 + 1;
                    current_cell_id = col + (row-1)*5;
                    parcel_idx = find(this.index_AMS == current_cell_id, 1);
                    
                    if ~isempty(parcel_idx)
                        obs(idx)   = this.x_boxes(parcel_idx)/l_max; 
                        obs(idx+1) = this.y_boxes(parcel_idx)/l_max; 
                        obs(idx+2) = this.d_boxes(parcel_idx)/(2 * this.d_AMS); 
                        obs(idx+3) = 1;                        
                    end
                end
            end
            
            Observation = obs; 
            for ii=1:this.max_gen_boxes
                if ii<=this.n_boxes_tot
                    % FIXED TRACKING DIRECTION
                    this.x_boxes_vect(this.cont, ii) = this.x_boxes(ii);
                    this.y_boxes_vect(this.cont, ii) = this.y_boxes(ii);
                    this.x_boxes_prec(ii) = this.x_boxes(ii);
                    this.y_boxes_prec(ii) = this.y_boxes(ii);
                else
                    this.x_boxes_vect(this.cont, ii) = -1;
                    this.y_boxes_vect(this.cont, ii) = -1;
                end
            end
            
            this.State = Observation;
            cont_exit = 0;
            for ii=1:this.n_boxes_tot
                if this.y_boxes(ii)>this.d_AMS*5
                    cont_exit = cont_exit + 1;
                    if this.pack_exited(ii) == 0
                        this.pack_exited(ii) = 1;
                        this.index_exit = this.index_exit + 1;
                        this.exit_order(this.index_exit) = ii;
                    end
                end
            end
            
            IsDone = (cont_exit == this.max_gen_boxes);
            this.IsDone = IsDone;
            
            Reward = getReward(this,AMS_actions);
            notifyEnvUpdated(this);
        end
        
        function InitialObservation = reset(this)
            this.index_exit = 0;
            this.pack_exited = zeros(10,1); 
            this.exit_order = zeros(10,1); 
            this.cont_new_box = 1;
            this.n_boxes_tot_vect = 2;
            this.n_boxes_tot = this.n_boxes_tot_vect;
            this.time = 0;
            this.vy_boxes = zeros(2,1);
            this.cont = 0;
            max_d_box = 2.*this.d_AMS; 
            min_d_box = 0.25*this.d_AMS; 
            l_AMS_matrix = this.d_AMS*this.n_j_AMS; 
            
            for ii=1:this.max_gen_boxes
                this.d_boxes(ii) = min_d_box + (max_d_box-min_d_box)*rand(1);
            end
            for ii=1:2
                if ii == 1
                    x1 = this.d_boxes(1)*0.5 + (this.d_AMS*2.5-this.d_boxes(1)*0.5)*rand(1);
                else
                    x2 = this.d_AMS*2.5+this.d_boxes(2)*0.5 + (l_AMS_matrix-(this.d_AMS*2.5+this.d_boxes(2)*0.5))*rand(1);
                    if abs(x2-x1) <= this.d_boxes(this.n_boxes_tot)/2+this.d_boxes(this.n_boxes_tot-1)/2
                        x2 = x1 + this.d_boxes(2)/2 + this.d_boxes(1)/2 + 0.1;
                    end
                    if x2-this.d_boxes(2)/2<0
                        x2 = this.d_boxes(2)/2;
                    elseif x2+this.d_boxes(2)/2>l_AMS_matrix
                        x2 = l_AMS_matrix-this.d_boxes(2)/2;
                    end
                    if abs(x2-x1) <= this.d_boxes(2)/2+this.d_boxes(1)/2
                        x1 = x2 - (this.d_boxes(2)/2 + this.d_boxes(1)/2 + 0.1);
                    end
                end
                y1 = 0.001 + rand(1)*0.01;
                y2 = 0.001 + rand(1)*0.05;
            end
            this.x_boxes(1) = x1;
            this.y_boxes(1) = y1;
            this.x_boxes(2) = x2;
            this.y_boxes(2) = y2;
            this.x_boxes_prec(1) = x1;
            this.x_boxes_prec(2) = x2;
            this.y_boxes_prec(1) = y1;
            this.y_boxes_prec(2) = y2;
            for ii=this.n_boxes_tot+1:this.max_gen_boxes
                this.x_boxes(ii) = 0;
                this.y_boxes(ii) = 0;
            end
            act_AMS = zeros(25,1);
            this.index_AMS = zeros(2,1);
            for ii=1:this.n_boxes_tot
                this.i_box(ii) = min(6, floor(this.y_boxes(ii) / this.d_AMS) + 1);
                if this.i_box(ii) == 6
                    this.j_box(ii) = 6;
                else
                    this.j_box(ii) = min(5, floor(this.x_boxes(ii) / this.d_AMS) + 1);
                end
                if this.i_box(ii)<6 && this.j_box(ii)<6
                    act_AMS(this.j_box(ii)+(this.i_box(ii)-1)*5) = 1;
                    this.index_AMS(ii) = this.j_box(ii)+(this.i_box(ii)-1)*5;
                else
                    this.index_AMS(ii) = 0;
                end
            end
            
            obs = zeros(100, 1);
            l_max = this.d_AMS * 5; 
            for row = 1:5
                for col = 1:5
                    idx = (col + (row-1)*5 - 1) * 4 + 1;
                    current_cell_id = col + (row-1)*5;
                    parcel_idx = find(this.index_AMS == current_cell_id, 1);
                    if ~isempty(parcel_idx)
                        obs(idx:idx+3) = [this.x_boxes(parcel_idx)/l_max; this.y_boxes(parcel_idx)/l_max; this.d_boxes(parcel_idx)/(2 * this.d_AMS); 1];
                    else
                        obs(idx:idx+3) = 0;
                    end
                end
            end
            InitialObservation = obs;
            this.State = InitialObservation;
            notifyEnvUpdated(this);
        end
    end
    
    methods  
        function varargout = plot(this)
            if isempty(this.Visualizer) || ~isvalid(this.Visualizer)
                this.Visualizer = AMSVisualizer_simple(this);
            else
                bringToFront(this.Visualizer);
            end
            if nargout
                varargout{1} = this.Visualizer;
            end
            this.VisualizeAnimation = true;
            this.VisualizeActions = false;
            this.VisualizeStates = false;
        end

    function Reward = getReward(this, AMS_actions)
        delta_y_total = 0;
        global_collision_count = 0;
        
        % 1. Evaluate Forward Progress & Check ALL global matrix grid collisions
        for ii = 1:this.n_boxes_tot
            if this.y_boxes(ii) > 0 && this.y_boxes(ii) <= 1.0
                % Guard against artificial spawn frame jumps
                if this.y_boxes_prec(ii) > 0.002
                    displacement = this.y_boxes(ii) - this.y_boxes_prec(ii); 
                    delta_y_total = delta_y_total + max(0, displacement);
                end
            end
            
            % Early Detection Guard: Scan for collisions over the entire sorting grid
            for jj = (ii+1):this.n_boxes_tot
                if this.y_boxes(ii) > 0 && this.y_boxes(jj) > 0
                    dist_centers = abs(this.y_boxes(ii) - this.y_boxes(jj));
                    min_safe_dist = (this.d_boxes(ii)/2) + (this.d_boxes(jj)/2);
                    if dist_centers < min_safe_dist
                        global_collision_count = global_collision_count + 1;
                    end
                end
            end
        end
        progress_reward = delta_y_total * 25; 
        
        % Define Singulation Corridor Zone
        area_start = 1.0; 
        area_end = 2.0; 
        in_zone = find(this.y_boxes > area_start & this.y_boxes <= area_end);
        K = length(in_zone); 
        
        live_bonus = 10.0; 
        
        if K < 2
            % Deduct points heavily if grid collisions happen even when exit corridor is empty
            Reward = progress_reward + live_bonus - (global_collision_count * 30); 
            return;
        end
        
        % Sort parcels sequentially along the axis of flow
        y_centers = this.y_boxes(in_zone);
        [~, sortIdx] = sort(y_centers);
        sorted_parcels = in_zone(sortIdx);
        
        ls = zeros(K-1, 1);
        all_gaps_positive = true;
        
        % 2. Evaluate Clear Separation (Singulation Corridor)
        for k = 1:K-1
            p_current = sorted_parcels(k);
            p_next = sorted_parcels(k+1);
            
            y_min_next = this.y_boxes(p_next) - (this.d_boxes(p_next)/2);
            y_max_curr = this.y_boxes(p_current) + (this.d_boxes(p_current)/2);
            
            dk = y_min_next - y_max_curr; 
            
            if dk <= 0
                all_gaps_positive = false;
                ls(k) = -30; % Severe pair penalty for overlapping at exit
            elseif dk < 0.05
                all_gaps_positive = false;
                ls(k) = 15 * (dk / 0.05); 
            else
                ls(k) = 15 + min(30 * (dk - 0.05), 15); 
            end
        end
        
        Reward = sum(ls) + progress_reward + live_bonus;
        
        % 3. Apply Multiplicative Global Penalties
        if global_collision_count > 0 || ~all_gaps_positive
            % Scale penalty relative to how many overlapping clusters are detected
            Reward = Reward - (50 + (global_collision_count * 20)); 
        end
        
        if all_gaps_positive && K >= 3 && global_collision_count == 0
            Reward = Reward + 30; % Pure, flawless singulation bonus
        end
    end
        
        function set.State(this,state)
            validateattributes(state,{'numeric'},{'finite','real','vector','numel',100},'','State');
            this.State = double(state(:));
            notifyEnvUpdated(this);
        end
    end 
        
    methods (Access = protected)
        % SLICING DIRECTION FIXED FOR MAXIMUM PERALLEL SPEED PERFORMANCE
        function info = getSimulationInfo(this)
            info.y_boxes_vect = this.y_boxes_vect(1:this.cont, :);
            info.x_boxes_vect = this.x_boxes_vect(1:this.cont, :);
            info.exit_order   = this.exit_order;
            info.index_exit   = this.index_exit;
        end
        function envUpdatedCallback(this)
        end
    end
end