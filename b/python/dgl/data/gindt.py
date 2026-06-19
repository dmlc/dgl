                self.graphs.append(g)
        self.labels = F.tensor(self.labels, F.int64)
        # if no attr
        if not self.nattrs_flag:
            if self.verbose:
                    g.ndata["attr"] = F.tensor(nattrs, F.float32)
                    self.nattrs_flag = True
                g.ndata["label"] = F.tensor(nlabels, F.int64)
                if len(self.nlabel_dict) > 1:
                    self.nlabels_flag = True
                    if tmp == len(nrow):
                        # no node attributes
                        nrow = [int(w) for w in nrow]
                    elif tmp < len(nrow):
                        # has node attributes
                        nrow = [int(w) for w in nrow[:tmp]]
                        nattr = [float(w) for w in nrow[tmp:]]
                        nattrs.append(nattr)
