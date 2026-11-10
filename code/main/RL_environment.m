classdef RL_environment < rl.env.MATLABEnvironment
    %AMS_RL: Template for defining custom environment in MATLAB.    
    
    %% Properties (set properties' attributes accordingly)
    properties
        % Specify and initialize environment's necessary properties
        % Velocity of the exit conveyor
        v_treadmill = 1.9; %  1.9 m/s, also change to 1.6 to speed up the training
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
        % Max number of parcels: episode contains at most 100 parcels.
        max_gen_boxes = 10;
        % Generation counter: counts parcel generation events.
        cont_new_box = 0;
        
        % Parcel geometry and trajectories
        % Diameter/size of each parcel.
        d_boxes; 
        % Current parcel coordinates.
        x_boxes;
        y_boxes; 
        
        % Full trajectory history FIXED to: [timesteps x parcels]
        x_boxes_vect; 
        y_boxes_vect; 
        
        % Timestep counter: Counts simulation steps.
        cont = 0;
        % Exit tracking: Has parcel exited?
        pack_exited;
        % Exit order
        exit_order;
        % Exit counter
        index_exit = 0;
        
        % Collision tolerance
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
            % 1. DEFINIZIONE DELLE INFO SULLE OSSERVAZIONI E AZIONI
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
            
            % 2. CHIAMATA OBBLIGATORIA ALLA SUPERCLASSE
            this = this@rl.env.MATLABEnvironment(ObservationInfo,ActionInfo);
            
            % 3. INIZIALIZZAZIONE DELLE PROPRIETÀ DIPENDENTI DA MAX_GEN_BOXES
            this.d_boxes = zeros(this.max_gen_boxes, 1);
            this.x_boxes = zeros(this.max_gen_boxes, 1);
            this.y_boxes = zeros(this.max_gen_boxes, 1); 
            this.pack_exited = zeros(this.max_gen_boxes, 1);
            this.exit_order = zeros(this.max_gen_boxes, 1);
            
            this.i_box = zeros(this.max_gen_boxes, 1);
            this.j_box = zeros(this.max_gen_boxes, 1);
            this.vy_boxes = zeros(this.max_gen_boxes, 1);
            this.x_boxes_prec = zeros(this.max_gen_boxes, 1);
            this.y_boxes_prec = zeros(this.max_gen_boxes, 1);
            
            % Spazio traiettorie aumentato per supportare l'orizzonte temporale denso
            this.x_boxes_vect = zeros(60000, this.max_gen_boxes);
            this.y_boxes_vect = zeros(60000, this.max_gen_boxes);
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
            
            % Tempo di generazione impostato a 0.5s
            if mod(round(this.time,2), 0.5) == 0 && this.n_boxes_tot<this.max_gen_boxes && this.time>0.25
                this.cont_new_box = this.cont_new_box + 1;
                gen_n_boxes = 2; % ==============================0
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
                        this.x_boxes_vect(this.cont-1, this.n_boxes_tot) = this.x_boxes(this.n_boxes_tot);
                        this.y_boxes_vect(this.cont-1, this.n_boxes_tot) = this.y_boxes(this.n_boxes_tot);
                    end
                end
            end
            
            % for ii=this.n_boxes_tot+1:this.max_gen_boxes
            %     this.x_boxes(ii) = -1;
            %     this.y_boxes(ii) = -1;
            % end
            % act_AMS = zeros(25,1);
            % this.index_AMS = zeros(this.max_gen_boxes, 1);
            % 
            % for ii=1:this.n_boxes_tot
            %     if this.x_boxes(ii)-this.d_boxes(ii)/2<0
            %         this.x_boxes(ii) = this.d_boxes(ii)/2;
            %     elseif this.x_boxes(ii)+this.d_boxes(ii)/2>l_AMS_matrix
            %         this.x_boxes(ii) = l_AMS_matrix-this.d_boxes(ii)/2;
            %     end
            % 
            %     this.i_box(ii) = min(6, floor(this.y_boxes(ii) / this.d_AMS) + 1);
            %     if this.i_box(ii) == 6
            %         this.j_box(ii) = 6;
            %     else
            %         this.j_box(ii) = min(5, floor(this.x_boxes(ii) / this.d_AMS) + 1);
            %     end
            % 
            %     if this.i_box(ii) < 6
            %         this.x_boxes(ii) = this.x_boxes(ii) + v_AMS(this.i_box(ii),this.j_box(ii))*this.dt*sin(rotation_AMS(this.i_box(ii),this.j_box(ii)));
            %         this.y_boxes(ii) = this.y_boxes(ii) + v_AMS(this.i_box(ii),this.j_box(ii))*this.dt*cos(rotation_AMS(this.i_box(ii),this.j_box(ii)));
            %         this.vy_boxes(ii) = v_AMS(this.i_box(ii),this.j_box(ii))*this.dt*cos(rotation_AMS(this.i_box(ii),this.j_box(ii)));
            %     else
            %         this.y_boxes(ii) = this.y_boxes(ii) + this.v_treadmill*this.dt;
            %         this.vy_boxes(ii) = this.v_treadmill;
            %     end
            % ==========================================================
        for ii = 1:this.n_boxes_tot
            % Correzione confini X sul bordo della matrice
            if this.x_boxes(ii) - this.d_boxes(ii)/2 < 0
                this.x_boxes(ii) = this.d_boxes(ii)/2;
            elseif this.x_boxes(ii) + this.d_boxes(ii)/2 > l_AMS_matrix
                this.x_boxes(ii) = l_AMS_matrix - this.d_boxes(ii)/2;
            end
            
            % --- CORREZIONE INDICI i_box e j_box ---
            % Usa max(1, ...) per evitare indici zero o negativi
            this.i_box(ii) = max(1, min(6, floor(this.y_boxes(ii) / this.d_AMS) + 1));
            
            if this.i_box(ii) == 6
                this.j_box(ii) = 6;
            else
                this.j_box(ii) = max(1, min(5, floor(this.x_boxes(ii) / this.d_AMS) + 1));
            end
            
            % --- MOVIMENTO PACCO IN SICUREZZA ---
            if this.i_box(ii) >= 1 && this.i_box(ii) <= 5 && this.j_box(ii) >= 1 && this.j_box(ii) <= 5
                % Pacco all'interno della griglia 5x5 AMS
                rot = rotation_AMS(this.i_box(ii), this.j_box(ii));
                v_cell = v_AMS(this.i_box(ii), this.j_box(ii));
                
                this.x_boxes(ii)  = this.x_boxes(ii)  + v_cell * this.dt * sin(rot);
                this.y_boxes(ii)  = this.y_boxes(ii)  + v_cell * this.dt * cos(rot);
                this.vy_boxes(ii) = v_cell * this.dt * cos(rot);
            else
                % Pacco oltre la griglia (sul nastro trasportatore di uscita)
                this.y_boxes(ii)  = this.y_boxes(ii) + this.v_treadmill * this.dt;
                this.vy_boxes(ii) = this.v_treadmill;
            end
    
    % ... (prosegui con il resto del codice delle collisioni)
            % ==========================================================
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
                    
                    parcel_indices = find(this.index_AMS == current_cell_id);
                    if ~isempty(parcel_indices)
                        [~, max_y_idx] = max(this.y_boxes(parcel_indices));
                        parcel_idx = parcel_indices(max_y_idx);
                        
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
                    this.x_boxes_vect(this.cont, ii) = this.x_boxes(ii);
                    this.y_boxes_vect(this.cont, ii) = this.y_boxes(ii);
                    this.x_boxes_prec(ii) = this.x_boxes(ii);
                    this.y_boxes_prec(ii) = this.y_boxes(ii);
                else
                    this.x_boxes_vect(this.cont, ii) = -1;
                    this.y_boxes_vect(this.cont, ii) = -1;
                end
            end
            
            this.State(:,1) = Observation;
            
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
            this.pack_exited = zeros(this.max_gen_boxes,1); 
            this.exit_order = zeros(this.max_gen_boxes,1); 
            this.cont_new_box = 1;
            this.n_boxes_tot_vect = 2;
            this.n_boxes_tot = 2; 
            this.time = 0;
            this.vy_boxes = zeros(this.max_gen_boxes,1); 
            this.cont = 0;
            max_d_box = 2.*this.d_AMS; 
            min_d_box = 0.25*this.d_AMS; 
            l_AMS_matrix = this.d_AMS*this.n_j_AMS; 
            
            for ii=1:this.max_gen_boxes
                this.d_boxes(ii) = min_d_box + (max_d_box-min_d_box)*rand(1);
            end
            
            x1 = this.d_boxes(1)*0.5 + (this.d_AMS*2.5-this.d_boxes(1)*0.5)*rand(1);
            x2 = this.d_AMS*2.5+this.d_boxes(2)*0.5 + (l_AMS_matrix-(this.d_AMS*2.5+this.d_boxes(2)*0.5))*rand(1);
            if abs(x2-x1) <= this.d_boxes(2)/2+this.d_boxes(1)/2
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
            y1 = 0.001 + rand(1)*0.01;
            y2 = 0.001 + rand(1)*0.05;
            
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
                this.x_boxes_prec(ii) = 0;
                this.y_boxes_prec(ii) = 0;
            end
            
            act_AMS = zeros(25,1);
            this.index_AMS = zeros(this.max_gen_boxes,1);
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
            this.State(:,1) = InitialObservation;
            notifyEnvUpdated(this);
        end
    end
    
    methods  
        function varargout = plot(this)
            if isempty(this.Visualizer) || ~isvalid(this.Visualizer)
                this.Visualizer = AMSVisualizer_simpleMODIFIED(this); 
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
        
        function Reward = getReward(this, ~)
            % GRUPPO DI PREMI STRUTTURATO PER ALTA DENSITÀ (0.5s)
            progress_reward = 0;
            collision_penalty = 0;
            
            % 1. Progresso fluido
            for ii = 1:this.n_boxes_tot
                if this.y_boxes(ii) > 0 && this.y_boxes(ii) <= 1.0
                    displacement = this.y_boxes(ii) - this.y_boxes_prec(ii); 
                    if displacement > 0
                        progress_reward = progress_reward + (displacement * 120); 
                    end
                end
            end
            
            % 2. Penalità di collisione CONTINUA
            for ii = 1:this.n_boxes_tot
                for jj = (ii+1):this.n_boxes_tot
                    if this.y_boxes(ii) > 0 && this.y_boxes(jj) > 0
                        dist_centers = abs(this.y_boxes(ii) - this.y_boxes(jj));
                        min_safe_dist = (this.d_boxes(ii)/2) + (this.d_boxes(jj)/2);
                        
                        if dist_centers < min_safe_dist
                            penetration = min_safe_dist - dist_centers;
                            collision_penalty = collision_penalty + (penetration * 250) + 8;
                        end
                    end
                end
            end
            
            % 3. Bonus di uscita
            exit_bonus = this.index_exit * 350; 
            
            Reward = progress_reward - collision_penalty + exit_bonus;
        end
        
        function set.State(this,state)
            validateattributes(state,{'numeric'},{'finite','real','vector','numel',100},'','State');
            this.State(:,1) = double(state(:));
            notifyEnvUpdated(this);
        end
    end 
        
    methods (Access = protected)
        function info = getSimulationInfo(this)
            info.y_boxes_vect = this.y_boxes_vect(1:this.cont, :);
            info.x_boxes_vect = this.x_boxes_vect(1:this.cont, :);
            info.exit_order   = this.exit_order;
            info.index_exit   = this.index_exit;
        end
        function envUpdatedCallback(~)
        end
    end
end